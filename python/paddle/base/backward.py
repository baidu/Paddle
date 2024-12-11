#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import collections
import copy
import logging
import os
import re
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, overload

import paddle.base

from . import core, framework, log_helper, unique_name
from .data_feeder import check_type
from .framework import program_guard
from .proto import framework_pb2

if TYPE_CHECKING:
    from collections.abc import Callable

    from paddle import Tensor
    from paddle.base.framework import Block
    from paddle.distributed.auto_parallel.static.dist_context import (
        DistributedContext,
    )

__all__ = []

_logger = log_helper.get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)


class ProgramStats:
    def __init__(self, block, ops):
        self.block = block
        self.ops = ops
        self.op_deps = {}  # op-> in_ops, out_ops
        self.var_op_deps = {}  # var as input op, var as output op

    def get_input_nodes(self):
        input_names = []
        for name in self.var_op_deps:
            if (
                len(self.var_op_deps[name]["var_as_output_ops"]) == 0
                and len(self.var_op_deps[name]["var_as_input_ops"]) > 0
            ):
                if self.block.var(name).persistable:
                    continue
                input_names.append(name)
        for op in self.ops:
            if op.desc.type() == "read":
                input_names.extend(op.desc.output_arg_names())
        return input_names

    def get_reserved_vars(self):
        var_name = []
        for op in self.ops:
            if op.desc.type() == "seed":
                var_name.extend(op.desc.output_arg_names())
        return var_name

    def get_out_of_subgraph_vars(self, begin_op_idx, end_op_idx):
        var_name = []
        for i in range(begin_op_idx, end_op_idx, 1):
            for name in self.ops[i].desc.output_arg_names():
                if name in self.var_op_deps:
                    for idx in self.var_op_deps[name]["var_as_input_ops"]:
                        if idx >= end_op_idx:
                            var_name.append(name)
            for name in self.ops[i].desc.input_arg_names():
                if name in self.var_op_deps:
                    for idx in self.var_op_deps[name]["var_as_output_ops"]:
                        if idx < begin_op_idx:
                            var_name.append(name)
        return var_name

    def is_subgraph(self, var_group1, var_group2):
        # should traverse from var_group1 to var_group2
        # max op idx in var_group2
        # min op idx in var_group1
        min_op_idx = len(self.ops)
        max_op_idx = -1
        for name in var_group1:
            if name not in self.var_op_deps:
                return False, min_op_idx, max_op_idx
        for name in var_group2:
            if name not in self.var_op_deps:
                return False, min_op_idx, max_op_idx
        for name in var_group1:
            op_idx = self.var_op_deps[name]["var_as_input_ops"]
            for idx in op_idx:
                min_op_idx = min(min_op_idx, idx)
        for name in var_group2:
            op_idx = self.var_op_deps[name]["var_as_output_ops"]
            for idx in op_idx:
                max_op_idx = max(max_op_idx, idx)
        if min_op_idx >= max_op_idx:
            return False, min_op_idx, max_op_idx

        return True, min_op_idx, max_op_idx

    def _update_segment_start(self, min_idx, pre_segment_end_idx):
        """
        persist vars of amp-related cast should be included in recompute segment
        """

        def is_amp_cast(op):
            return (
                op.desc.type() == 'cast'
                and self.block.var(op.desc.input_arg_names()[0]).persistable
            )

        idx_ = min_idx - 1
        updated_min_idx = min_idx
        while idx_ > pre_segment_end_idx:
            if is_amp_cast(self.ops[idx_]):
                _logger.info(
                    f"found amp-cast op: {self.ops[idx_].desc.type()}, : {self.ops[idx_].desc.input_arg_names()[0]}"
                )
                updated_min_idx = idx_
                idx_ -= 1
            else:
                break

        return updated_min_idx

    def build_stats(self):
        for i, op in enumerate(self.ops):
            self.op_deps[i] = {"in_ops": [], "out_ops": []}
            for j, name in enumerate(op.desc.input_arg_names()):
                if name in self.var_op_deps:
                    self.op_deps[i]["in_ops"].extend(
                        self.var_op_deps[name]["var_as_output_ops"]
                    )
            for j, name in enumerate(op.desc.input_arg_names()):
                if name in self.var_op_deps:
                    self.var_op_deps[name]["var_as_input_ops"].extend([i])
                else:
                    self.var_op_deps[name] = {}
                    self.var_op_deps[name]["var_as_input_ops"] = [i]
                    self.var_op_deps[name]["var_as_output_ops"] = []

            for j, name in enumerate(op.desc.output_arg_names()):
                if name in self.var_op_deps:
                    self.var_op_deps[name]["var_as_output_ops"].extend([i])
                else:
                    self.var_op_deps[name] = {}
                    self.var_op_deps[name]["var_as_input_ops"] = []
                    self.var_op_deps[name]["var_as_output_ops"] = [i]

            for op_idx in self.op_deps[i]["in_ops"]:
                self.op_deps[op_idx]["out_ops"].extend([i])

    def sort_checkpoints(self, checkpoints_name):
        sorted_checkpoints = []
        for name in checkpoints_name:
            if name not in self.var_op_deps:
                _logger.info(
                    f"Recompute Optimizer: deleted {name} from checkpoints, because it is not used in paddle program."
                )
            elif self.var_op_deps[name]["var_as_output_ops"] == []:
                # input nodes
                sorted_checkpoints.append((name, -1))
            else:
                sorted_checkpoints.append(
                    (name, max(self.var_op_deps[name]["var_as_output_ops"]))
                )
        sorted_checkpoints = sorted(sorted_checkpoints, key=lambda x: x[1])
        return [x[0] for x in sorted_checkpoints]

    def modify_forward_desc_for_recompute(self):
        op_types = [op.desc.type() for op in self.ops]
        if "dropout" not in op_types:
            return

        op_idx = 0
        while op_idx < len(self.ops):
            op = self.ops[op_idx]
            if op.desc.type() != "dropout":
                op_idx += 1
                continue
            # already insert seed op before dropout
            if op.input('Seed') is not None and len(op.input('Seed')) == 1:
                op_idx += 1
                continue
            # add a seed op so that the two dropout op can generate same output
            op_unique_name = unique_name.generate("seed")
            var_unique_name = unique_name.generate_with_ignorable_key(
                ".".join([op_unique_name, 'tmp'])
            )
            added_var = self.block.create_var(
                name=var_unique_name,
                dtype='int32',
                type=core.VarDesc.VarType.DENSE_TENSOR,
                persistable=False,
                stop_gradient=False,
            )
            seed = 0 if op.attr("fix_seed") is False else int(op.attr("seed"))

            op_device_attr_name = (
                core.op_proto_and_checker_maker.kOpDeviceAttrName()
            )
            op_device = ""
            if op.desc.has_attr(op_device_attr_name):
                op_device = op.desc.attr(op_device_attr_name)

            # Setting the force_cpu of seed to true will make the output of seed in cpu memory,
            # reduce the synchronous copy from GPU to CPU in dropout, and reduce the communication hang
            added_op = self.block._insert_op(
                index=op.idx,
                type='seed',
                inputs={},
                outputs={'Out': [added_var]},
                attrs={'seed': seed, 'op_device': op_device, 'force_cpu': True},
            )
            self.ops.insert(op_idx, added_op)
            # modify dropout op desc so that it accept a seed var as input
            op.desc.set_input("Seed", [var_unique_name])
            op.desc.remove_attr("fix_seed")
            op.desc.remove_attr("seed")
            self.block._sync_with_cpp()
            op_idx += 2


def _pretty_op_desc_(op_desc, prefix):
    out_s = "{}\tname:[{}]\n{}    \tinputs:[{}]\n{}    \toutputs:[{}]".format(
        prefix + "_op",
        str(op_desc.type()),
        prefix + "_input",
        " ".join(op_desc.input_arg_names()),
        prefix + "_output",
        " ".join(op_desc.output_arg_names()),
    )
    return out_s


def _add_needed_descs_to_block(
    descs, block, main_block, in_memory_vars, grad_op_id_to_fwd_op=None
):
    if len(descs) == 0:
        return []
    result_descs = []
    op_role_attr_name = core.op_proto_and_checker_maker.kOpRoleAttrName()
    backward = core.op_proto_and_checker_maker.OpRole.Backward
    for desc in descs:
        origin_desc = desc
        origin_is_operator = False
        if isinstance(desc, framework.Operator):
            desc = desc.desc
            origin_is_operator = True
        if isinstance(desc, tuple):
            desc = desc[0]
        is_needed = False
        for name in desc.output_arg_names():
            if main_block.has_var(name) and main_block.var(name).persistable:
                continue
            if name not in in_memory_vars:
                is_needed = True
        if is_needed:
            if origin_is_operator and grad_op_id_to_fwd_op is not None:
                grad_op_id_to_fwd_op[desc.original_id()] = origin_desc
            new_op_desc = block.desc.append_op()
            new_op_desc.copy_from(desc)
            new_op_desc._set_attr(op_role_attr_name, backward)
            if desc.has_attr('op_device'):
                new_op_desc._set_attr('op_device', desc.attr('op_device'))
            result_descs.append(new_op_desc)
    return result_descs


def _add_descs_to_block(descs, block, grad_op_id_to_fwd_op=None):
    if len(descs) == 0:
        return []
    result_descs = []
    op_role_attr_name = core.op_proto_and_checker_maker.kOpRoleAttrName()
    backward = core.op_proto_and_checker_maker.OpRole.Backward
    for desc in descs:
        if isinstance(desc, framework.Operator):
            # for recompute, should record recompute ops
            if grad_op_id_to_fwd_op is not None:
                grad_op_id_to_fwd_op[desc.desc.original_id()] = desc
            desc = desc.desc
        if isinstance(desc, tuple):
            desc = desc[0]
        new_op_desc = block.desc.append_op()
        new_op_desc.copy_from(desc)
        new_op_desc._set_attr(op_role_attr_name, backward)
        if desc.has_attr('op_device'):
            new_op_desc._set_attr('op_device', desc.attr('op_device'))
        result_descs.append(new_op_desc)
    return result_descs


def _find_loss_op_(loss):
    for op in reversed(loss.block.ops):
        assert isinstance(op, framework.Operator)
        if (
            len(op.output_arg_names) == 1
            and op.output_arg_names[0] == loss.name
        ):
            loss.op = op
            break
    if loss.op is None:
        raise ValueError("loss.op is None. Should not happen")


def _rename_arg_(op_descs, old_name, new_name, begin_idx=None, end_idx=None):
    """
    Traverse all ops in op_descs[begin_idx : end_idx],
    if any op has inputs/outputs named "old_name", rename it as 'new_name'
    """
    if begin_idx is None:
        begin_idx = 0
    if end_idx is None:
        end_idx = len(op_descs)
    if isinstance(op_descs, (list, tuple)):
        for i in range(begin_idx, end_idx):
            op_desc = op_descs[i]
            if isinstance(op_desc, tuple):
                op_desc = op_desc[0]
            op_desc._rename_input(old_name, new_name)
            op_desc._rename_output(old_name, new_name)
    if isinstance(op_descs, collections.OrderedDict):
        for key, value in op_descs.items():
            if isinstance(value, (list, tuple)):
                for op_desc in value:
                    op_desc._rename_input(old_name, new_name)
                    op_desc._rename_output(old_name, new_name)


def _create_op_desc_(op_type, inputs, outputs, attrs):
    """
    Create a C++ OpDesc object with specified inputs, outputs and attributes.
    """
    op_desc = core.OpDesc()
    op_desc.set_type(op_type)
    for para, args in inputs.items():
        op_desc.set_input(
            para,
            [arg.decode() if isinstance(arg, bytes) else arg for arg in args],
        )
    for para, args in outputs.items():
        op_desc.set_output(
            para,
            [arg.decode() if isinstance(arg, bytes) else arg for arg in args],
        )
    op_role_attr_name = core.op_proto_and_checker_maker.kOpRoleAttrName()
    op_device_attr_name = core.op_proto_and_checker_maker.kOpDeviceAttrName()

    if op_role_attr_name not in attrs:
        attrs[op_role_attr_name] = (
            core.op_proto_and_checker_maker.OpRole.Backward
        )
    if op_device_attr_name not in attrs:
        attrs[op_device_attr_name] = ""
    for name, val in attrs.items():
        if isinstance(val, framework.Block):
            op_desc.set_block_attr(name, val.desc)
        else:
            op_desc._set_attr(name, val)
    return op_desc


def _create_loss_op_desc_(loss):
    # 0-D Tensor or 0-Size Tensor
    if len(loss.shape) == 0 or 0 in loss.shape:
        create_shape = loss.shape
    else:
        create_shape = [1]
    op_desc = _create_op_desc_(
        "fill_constant",
        {},
        {"Out": [_append_grad_suffix_(loss.name)]},
        {
            "shape": create_shape,
            "value": 1.0,
            "dtype": loss.dtype,
            "force_cpu": False,
            core.op_proto_and_checker_maker.kOpRoleAttrName(): int(
                core.op_proto_and_checker_maker.OpRole.Backward
            )
            | int(core.op_proto_and_checker_maker.OpRole.Loss),
            core.op_proto_and_checker_maker.kOpDeviceAttrName(): loss.op.attr(
                core.op_proto_and_checker_maker.kOpDeviceAttrName()
            ),
        },
    )
    return op_desc


def _infer_var_data_type_shape_(grad_var_name, block):
    """
    Infer the data type and shape of given grad variable
    """
    grad_var = block.desc.find_var(grad_var_name.encode())
    fwd_name = _strip_grad_suffix_(grad_var_name)
    if block.desc.has_var_recursive(fwd_name.encode()):
        fwd_var = block.desc.find_var_recursive(fwd_name.encode())
        grad_var.set_dtype(fwd_var.dtype())
        grad_var.set_shape(fwd_var.shape())
    else:
        # TODO(jiabin): Maybe we should not to this to cause some unexpected error on dtype
        warnings.warn(
            f"Set grad var: {grad_var_name} dtype to default FP32, since we can't find its related forward var"
        )
        grad_var.set_dtype(core.VarDesc.VarType.FP32)


def _all_in_set_(cands, s):
    """
    Test if all elements of 'cands' are in set 's'
    """
    if len(cands) == 0:
        return False
    for c in cands:
        if c not in s:
            return False
    return True


def _some_in_set_(cands, s):
    """
    Test if some elements of 'cands' are in set 's'
    """
    if len(cands) == 0:
        return False
    for c in cands:
        if c in s:
            return True
    return False


def _strip_grad_suffix_(name):
    """
    Strip the grad suffix from the given variable name
    e.g. x@GRAD ==> x
         x@GRAD@GRAD ==> x
         y@GRAD@RENAME@1 ==> y
         z@GRAD_slice_0@GRAD ==> z@GRAD_slice_0
         grad/grad/z@GRAD@RENAME@block0@1@GRAD ==> z
    """
    pos = re.search(f'{core.grad_var_suffix()}+@', name) or re.search(
        f'{core.grad_var_suffix()}$', name
    )
    new_name = name[: pos.start()] if pos is not None else name
    new_pos = name.rfind('grad/')
    return new_name[new_pos + 5 :] if new_pos != -1 else new_name


def _append_grad_suffix_(name):
    """
    Append grad suffix to the given variable name
    e.g. x ==> x@GRAD
    """
    return name + core.grad_var_suffix()


def _accumulate_gradients_by_sum_op_(
    var_name, renamed_vars, pending_sum_ops, op_idx, op_device=""
):
    """
    Use sum op to accumulate_gradients, the gradients are stored in renamed_vars.
    """
    if op_idx not in pending_sum_ops.keys():
        pending_sum_ops[op_idx] = []
    pending_sum_ops[op_idx].append(
        _create_op_desc_(
            "sum",
            {"X": renamed_vars[var_name]},
            {"Out": [var_name]},
            {"op_device": op_device},
        )
    )
    renamed_vars[var_name] = [var_name]


def _accumulate_gradients_by_add_ops_(
    var_name,
    renamed_vars,
    pending_sum_ops,
    op_idx,
    op_device="",
    grad_var_to_var=None,
):
    """
    Use several inplace add op to accumulate_gradients, the gradients are stored in renamed_vars.
    """
    if op_idx not in pending_sum_ops.keys():
        pending_sum_ops[op_idx] = []
    out_name = renamed_vars[var_name][0]
    for i in range(1, len(renamed_vars[var_name])):
        x_name = out_name
        y_name = renamed_vars[var_name][i]
        if i != len(renamed_vars[var_name]) - 1:
            out_name = var_name + '@ADD@' + str(i)
        else:
            out_name = var_name
        pending_sum_ops[op_idx].append(
            _create_op_desc_(
                "grad_add",
                {"X": [x_name], "Y": [y_name]},
                {"Out": [out_name]},
                {"op_device": op_device},
            )
        )
        # record mapping between out grad var name and fwd var name (only for auto parallel)
        if grad_var_to_var is not None:
            if var_name in grad_var_to_var:
                grad_var_to_var[out_name] = grad_var_to_var[var_name]
            else:
                grad_var_to_var[out_name] = var_name
    renamed_vars[var_name] = [var_name]


def _addup_repetitive_outputs_(
    op_descs,
    block_idx,
    grad_var_to_var=None,
    grad_op_id_to_fwd_op=None,
    topo_order_for_backward=None,
):
    """
    In backward part, an variable may be the output of more than one ops.
    And one op may yield its multiple outputs to the same variable.
    In these cases, the variable should be the accumulation of all the outputs.
    `sum_op`s are added to implement the accumulate.

    Args:
        grad_var_to_var(dict): used to build the mapping between grad var name and forward var name.
        Only for auto parallel.
    """

    _MAX_ADD_NUM_ = framework._global_flags()['FLAGS_max_inplace_grad_add']
    topo_order_for_grad_name = {}
    # pending_sum_ops = []
    pending_sum_ops = collections.OrderedDict()
    var_rename_count = collections.defaultdict(int)
    renamed_vars = collections.defaultdict(list)
    renamed_var_start_idx = collections.defaultdict(list)
    var_device = collections.defaultdict(str)

    def _change_order_by_topo_order(var_name):
        if topo_order_for_backward is None:
            return
        origin_names = renamed_vars[var_name]
        origin_names.sort(key=lambda x: topo_order_for_grad_name[x])

    for idx, op_desc in enumerate(op_descs):
        op_device_attr_name = (
            core.op_proto_and_checker_maker.kOpDeviceAttrName()
        )
        op_device = ""
        if op_desc.has_attr(op_device_attr_name):
            op_device = op_desc.attr(op_device_attr_name)
        for var_name in op_desc.input_arg_names():
            if "@GRAD" not in var_name:
                continue
            if len(renamed_vars[var_name]) > 1:
                if len(renamed_vars[var_name]) > _MAX_ADD_NUM_:
                    _change_order_by_topo_order(var_name)
                    _accumulate_gradients_by_sum_op_(
                        var_name,
                        renamed_vars,
                        pending_sum_ops,
                        idx,
                        var_device[var_name],
                    )
                else:
                    _change_order_by_topo_order(var_name)
                    _accumulate_gradients_by_add_ops_(
                        var_name,
                        renamed_vars,
                        pending_sum_ops,
                        idx,
                        var_device[var_name],
                        grad_var_to_var,
                    )

        for param_idx, param_name in enumerate(op_desc.output_names()):
            arg_names = op_desc.output(param_name)
            for arg_idx, var_name in enumerate(arg_names):
                if "@GRAD" not in var_name:
                    continue
                # if "@RENAME@" in var_name:
                #    continue
                if (
                    var_name == core.empty_var_name()
                    or var_name in op_desc.input_arg_names()
                ):
                    # empty variable or inplace op
                    continue
                if len(renamed_vars[var_name]) == 0:
                    # it's the first time we get the variable
                    renamed_vars[var_name] = [var_name]
                    renamed_var_start_idx[var_name] = idx
                    topo_order_for_grad_name[var_name] = (
                        topo_order_for_backward[op_desc]
                        if topo_order_for_backward
                        and op_desc in topo_order_for_backward
                        else 1
                    )
                else:
                    if len(renamed_vars[var_name]) == 1:
                        new_name = (
                            var_name
                            + "@RENAME@block"
                            + str(block_idx)
                            + "@"
                            + str(var_rename_count[var_name])
                        )
                        var_rename_count[var_name] += 1
                        # Build the mapping between the new_name and var_name (Only for auto parallel)
                        if grad_var_to_var is not None:
                            if var_name in grad_var_to_var:
                                grad_var_to_var[new_name] = grad_var_to_var[
                                    var_name
                                ]
                            else:
                                grad_var_to_var[new_name] = var_name
                        # rename original var_name
                        topo_order_for_grad_name[new_name] = (
                            topo_order_for_grad_name[var_name]
                        )
                        renamed_vars[var_name][0] = new_name
                        # before change: _rename_arg_(op_descs, var_name,
                        #                             new_name, 0, idx)
                        # rename arg from idx of the first appearance
                        # in backward, not always from 0
                        _rename_arg_(
                            op_descs,
                            var_name,
                            new_name,
                            renamed_var_start_idx[var_name],
                            idx,
                        )
                        _rename_arg_(pending_sum_ops, var_name, new_name)

                        for p in op_desc.output_names()[:param_idx]:
                            p_arg_names = op_desc.output(p)
                            if var_name in p_arg_names:
                                op_desc.set_output(
                                    p,
                                    [
                                        new_name if x == var_name else x
                                        for x in p_arg_names
                                    ],
                                )

                        arg_names = [
                            new_name if x == var_name else x
                            for x in arg_names[:arg_idx]
                        ] + arg_names[arg_idx:]

                    new_name = (
                        var_name
                        + "@RENAME@block"
                        + str(block_idx)
                        + "@"
                        + str(var_rename_count[var_name])
                    )
                    var_rename_count[var_name] += 1
                    # Build the mapping between the new_name and var_name (Only for auto parallel)
                    if grad_var_to_var is not None:
                        if var_name in grad_var_to_var:
                            grad_var_to_var[new_name] = grad_var_to_var[
                                var_name
                            ]
                        else:
                            grad_var_to_var[new_name] = var_name
                    arg_names[arg_idx] = new_name
                    op_desc.set_output(param_name, arg_names)
                    renamed_vars[var_name].append(new_name)
                    # record the latest device
                    var_device[var_name] = op_device
                    topo_order_for_grad_name[new_name] = (
                        topo_order_for_backward[op_desc]
                        if topo_order_for_backward
                        and op_desc in topo_order_for_backward
                        else 1
                    )

    for var_name, inputs in renamed_vars.items():
        if len(renamed_vars[var_name]) > 1:
            if len(renamed_vars[var_name]) > _MAX_ADD_NUM_:
                _change_order_by_topo_order(var_name)
                _accumulate_gradients_by_sum_op_(
                    var_name,
                    renamed_vars,
                    pending_sum_ops,
                    len(op_descs),
                    var_device[var_name],
                )
            else:
                _change_order_by_topo_order(var_name)
                _accumulate_gradients_by_add_ops_(
                    var_name,
                    renamed_vars,
                    pending_sum_ops,
                    len(op_descs),
                    var_device[var_name],
                )

    op_descs_len = len(op_descs)
    # sum_op descs are sorted according to their insert position
    for key, value in collections.OrderedDict(
        reversed(list(pending_sum_ops.items()))
    ).items():
        # NOTE(zhiqiu): Since reversed, the idx of op_descs to be inserted will remains correct.
        # For example, [0, 1, 2], and we want to insert 'a' at idx 1, 'b' at idx 2, and the expected result is [0, 1, 'a', 2, 'b'].
        # If reversed, we first insert 'b' at idx 2, it becomes [0, 1, 2, 'b'], and then insert 'a' at idx 1, it becomes [0, 1, 'a', 2, 'b'].
        # If not reverse, we first insert 'a' at idx 1, it becomes [0, 1, 'a', 2], and then insert 'b' at idx 2, it becomes [0, 1, 'a', 'b', 2].
        idx = key
        for i, op in enumerate(value):
            # update the mapping between fwd and bwd
            target_idx = idx - 1 if idx == op_descs_len else idx + i
            if (
                grad_op_id_to_fwd_op is not None
                and grad_op_id_to_fwd_op.get(
                    op_descs[target_idx].original_id(), None
                )
                is not None
            ):
                grad_op_id_to_fwd_op[op.original_id()] = grad_op_id_to_fwd_op[
                    op_descs[target_idx].original_id()
                ]
            op_descs.insert(idx + i, op)

    return op_descs


def _remove_no_grad_branch_(
    op_descs, no_grad_set, grad_op_id_to_fwd_op=None, target_vars=[]
):
    """
    Remove unnecessary grad ops
    A grad op can be removed in two cases:
        1. all outputs of the grad op are in 'no_grad_set'
        2. all grad inputs of the grad op are in 'no_grad_set'
    NOTE: we will skip target_vars's grad name.
    """

    def _op_can_be_removed_(op_desc, no_grad_set):
        out_arg_names = op_desc.output_arg_names()
        if len(out_arg_names) == 0 or _all_in_set_(out_arg_names, no_grad_set):
            return True
        if _all_in_set_(
            [
                name
                for name in op_desc.input_arg_names()
                if name.find(core.grad_var_suffix()) != -1
            ],
            no_grad_set,
        ):
            no_grad_set.update(set(out_arg_names) - target_grad_var_names)
            return True
        return False

    # Remove ops whose outputs are all in no_grad_dict
    target_grad_var_names = {
        var.name + core.grad_var_suffix() for var in target_vars
    }
    op_descs = [
        op_desc
        for op_desc in op_descs
        if not _op_can_be_removed_(op_desc, no_grad_set)
    ]
    # Insert fill_any_like_op with value 0
    to_insert = []
    if not core._is_bwd_prim_enabled():
        for idx, op_desc in enumerate(op_descs):
            for arg in op_desc.input_arg_names():
                # arg is a gradient var name and arg should not have gradient
                if core.grad_var_suffix() in arg and arg in no_grad_set:
                    x_in = _strip_grad_suffix_(arg)
                    # the reason should be: arg can be input of another grad op
                    # and the op is a not-to-remove op
                    new_op_desc = _create_op_desc_(
                        "fill_any_like",
                        {"X": [x_in]},
                        {"Out": [arg]},
                        {'value': 0, 'dtype': -1},
                    )
                    # update the mapping between fwd and bwd
                    if (
                        grad_op_id_to_fwd_op is not None
                        and grad_op_id_to_fwd_op.get(
                            op_desc.original_id(), None
                        )
                        is not None
                    ):
                        grad_op_id_to_fwd_op[new_op_desc.original_id()] = (
                            grad_op_id_to_fwd_op[op_desc.original_id()]
                        )
                    to_insert.append((new_op_desc, idx))

    [op_descs.insert(p[1], p[0]) for p in reversed(to_insert)]

    return op_descs


def _find_not_need_ops(grad_op_descs, forward_ops, input_grad_names_set):
    """
    Pruning Program with Structural Analysis Method of Computational Graph.
    The nodes of the computational graph composed of backward OPS should be
    interconnected. If there are unconnected sub-graphs in the computational graph,
    these sub-graphs should be cut off.

    Args:
        grad_op_descs(list[core.OpDesc]): The candidate backward OpDescs.
        forward_ops(list[Operator]): The forward ops.
        input_grad_names_set(set): this set is used to store the gradients' name
            which is generated by backward ops, and input_grad_names_set can help
            to prune the unnecessary backward ops.

    Return:
        (set[core.OpDesc]): A set of OpDescs which should be pruned.
    """

    class Var:
        def __init__(self, var_name):
            self.var_name = var_name
            self.gen_op = None
            self.pending_ops = []

        def set_gen_op(self, gen_op):
            assert isinstance(gen_op, Op)
            assert self.gen_op is None
            self.gen_op = gen_op

        def add_pending_op(self, op):
            assert isinstance(op, Op)
            self.pending_ops.append(op)

    class Op:
        def __init__(self, op_desc):
            self.op_desc = op_desc
            self.inputs = []
            self.outputs = []

        def insert_input(self, var):
            assert isinstance(var, Var)
            self.inputs.append(var)

        def insert_output(self, var):
            assert isinstance(var, Var)
            self.outputs.append(var)

    var_versions = {}

    def _create_node(name):
        if name not in var_versions.keys():
            var_versions[name] = [Var(name)]
        else:
            var_versions[name].append(Var(name))
        return var_versions[name][-1]

    def _create_or_get_last_version_node(name):
        if name not in var_versions.keys():
            var_versions[name] = [Var(name)]
        return var_versions[name][-1]

    def _create_op_node(op_desc):
        op_node = Op(op_desc)
        for input in op_desc.input_arg_names():
            var = _create_or_get_last_version_node(name=input)
            var.add_pending_op(op_node)
            op_node.insert_input(var)
        for output in op_desc.output_arg_names():
            var = _create_node(name=output)
            var.set_gen_op(op_node)
            op_node.insert_output(var)
        return op_node

    # Record the forward vars
    forward_vars_set = (
        set() if input_grad_names_set is None else set(input_grad_names_set)
    )
    for op in forward_ops:
        forward_vars_set.update(op.desc.input_arg_names())
        forward_vars_set.update(op.desc.output_arg_names())

    # Record the vars which are created during backward and is not generated by op.
    backward_vars_set = set()
    # special_op_nodes is the candidate sub-graph head node.
    special_op_nodes = set()
    for op_desc in grad_op_descs:
        input_set = set(op_desc.input_arg_names())
        # The new_vars are created during backward and is not generated by op.
        new_vars = input_set - forward_vars_set - backward_vars_set
        backward_vars_set.update(op_desc.output_arg_names())

        op_node = _create_op_node(op_desc)
        if len(new_vars) == len(input_set):
            special_op_nodes.add(op_node)

    not_need_op_descs = []
    # Start traversing all candidate sub-graph headers to check whether
    # they are connected to backward computational graphs, and if they are
    # not, list them in not_need_op_descs
    for special_op_node in special_op_nodes:
        op_list = [special_op_node]
        ready_vars = set(special_op_node.inputs)
        remove_ops = True
        candidate_ops = [special_op_node]
        while len(candidate_ops) > 0:
            op_node = candidate_ops.pop(0)
            if _all_in_set_(op_node.inputs, ready_vars):
                for out_var in op_node.outputs:
                    candidate_ops.extend(out_var.pending_ops)
                    op_list.extend(out_var.pending_ops)
                ready_vars.update(op_node.outputs)
            else:
                remove_ops = False
                break
        if remove_ops:
            not_need_op_descs.extend([node.op_desc for node in op_list])
    not_need_op_descs_set = set(not_need_op_descs)
    grad_op_descs_set = set(grad_op_descs)
    # If a backward computational graph is simply one sub-graph header, the
    # not_need_op_descs will be whole graph, this IF clause avoids it.
    if grad_op_descs_set == not_need_op_descs_set:
        return set()
    return not_need_op_descs_set


def serialize_op_decs(op_desc):
    protostr = op_desc.serialize_to_string()
    proto = framework_pb2.OpDesc.FromString(bytes(protostr))
    return proto.__str__()


def _append_backward_ops_with_checkpoints_(
    block,
    ops,
    target_vars,
    target_block,
    no_grad_dict,
    grad_to_var,
    checkpoints,
    grad_op_id_to_fwd_op=None,
):
    """
    Create grad ops with forward ops, and insert them into given block

    Args:
        block(Block): the block where forward ops are
        ops(Op): the forward operators whose forward recomputation backward ops need to be added
        target_vars(list[Tensor]): the loss vars we want to calculate gradient.
        target_block(Block): the block which is going to hold new generated grad ops
        no_grad_dict(dict):
            key(int) block index
            val(str): corresponding forward variable name
        checkpoints: variables that a user defined as checkpoint for forward recomputation

    Algorithms:
        0) deal with forward recomputing program descs
        1) find ops between checkpoints, i.e. recompute_segments
        2) go through all forward ops and induct all variables that will be hold in memory
            a. variables that are used across segments will be held in memory
            b. output of dropout op will be held in memory
            c. input variables will be held in memory
        3) go through each recompute_segments, add backward ops with forward recomputation
            a. add ops in current recompute_segment as forward recomputation ops
            b. rename all non-checkpoint variables in recomputation ops
            c. add backward ops of current recomputation ops
            d. add sum op for repetitive_outputs
        4) remove no grad branch as it is in _remove_no_grad_branch_
        5) Note1: all appended ops' OpRole are Backward
        6) Note2: all variables with new name should be returned so that _append_backward_vars_ can be called
        7) Note3: current forward recomputation backpropagation does not handle programs with subblock
    """

    checkpoints_name = [x.name for x in checkpoints]
    checkpoints_name = list(set(checkpoints_name))
    local_block = block.program._create_block()
    buffer_block = block.program._create_block()
    # 0) deal with forward recomputing program descs
    program_stat = ProgramStats(block, ops)
    program_stat.modify_forward_desc_for_recompute()
    program_stat.build_stats()

    # 1) find ops between checkpoints, i.e. recompute_segments
    checkpoints_name = program_stat.sort_checkpoints(checkpoints_name)
    segments = []

    if len(checkpoints_name) == 1:
        # only one checkpoint
        max_op_idx = -1
        var_group = [checkpoints_name[0]]
        for name in var_group:
            if name not in program_stat.var_op_deps:
                break
            op_idx = program_stat.var_op_deps[name]["var_as_output_ops"]
            # only count the last generate op
            for idx in op_idx:
                max_op_idx = max(max_op_idx, idx)
        if max_op_idx > 0:
            segments.append([0, max_op_idx + 1])
    else:
        start_idx = 0
        pre_segment_end_idx = -1
        while True:
            if start_idx >= len(checkpoints_name) - 1:
                break
            # min_idx: checkpoint_1' s input op
            # max_idx: checkpoint_2' s output op
            flag, min_idx, max_idx = program_stat.is_subgraph(
                [checkpoints_name[start_idx]], [checkpoints_name[start_idx + 1]]
            )
            if flag:
                # max_idx + 1 since the exact and used segment end idx is max_idx
                min_idx = program_stat._update_segment_start(
                    min_idx, pre_segment_end_idx
                )
                segments.append([min_idx, max_idx + 1])
            else:
                _logger.info(
                    f"Could not recompute op range [{min_idx}] - [{max_idx + 1}] "
                )

            start_idx += 1

    if segments != [] and segments[0][0] != 0:
        recompute_segments = [[0, segments[0][0]], *segments]
    else:
        recompute_segments = segments

    for i, (idx1, idx2) in enumerate(recompute_segments):
        _logger.info(f"recompute segment[{i}]")
        _logger.info(
            f"segment start op: [{ops[idx1].desc.type()}]: [{ops[idx1].desc.input_arg_names()}]"
        )
        _logger.info(
            f"segment end op: [{ops[idx2 - 1].desc.type()}]: [{ops[idx2 - 1].desc.input_arg_names()}]"
        )
        _logger.info(f"recompute segment[{i}]")
        _logger.info(
            f"segment start op: [{ops[idx1].desc.type()}]: [{ops[idx1].desc.input_arg_names()}]"
        )
        _logger.info(
            f"segment end op: [{ops[idx2 - 1].desc.type()}]: [{ops[idx2 - 1].desc.input_arg_names()}]"
        )

    # 2) go through all forward ops and induct all variables that will be hold in memory
    vars_should_be_hold = []
    # a. variables that are used across segments will be held in memory
    for segment in recompute_segments:
        vars_should_be_hold.extend(
            program_stat.get_out_of_subgraph_vars(segment[0], segment[1])
        )

    cross_vars = set(vars_should_be_hold) - set(checkpoints_name)
    _logger.info(
        f"found [{len(cross_vars)}] vars which cross recompute segment: [{cross_vars}], better checkpoints might be set to reduce those vars"
    )

    # b. output of seed op should be kept in memory
    vars_should_be_hold.extend(program_stat.get_reserved_vars())
    # c. input variables are checkpoints
    vars_should_be_hold.extend(program_stat.get_input_nodes())
    vars_should_be_hold = list(set(vars_should_be_hold))

    # 3) go through each recompute_segments, add backward ops with forward recomputation
    grad_op_descs = []
    var_name_dict = {}

    vars_in_memory = vars_should_be_hold + checkpoints_name

    max_calculated_op_position = len(ops)
    device_attr_name = core.op_proto_and_checker_maker.kOpDeviceAttrName()
    if recompute_segments == []:
        gap_ops = ops[0:max_calculated_op_position]
        for op in reversed(gap_ops):
            if op.has_attr("sub_block"):
                raise Exception(
                    "Recompute don't support ops with sub_block"
                    "invoke op: {}".format(
                        _pretty_op_desc_(op.desc, "with_sub_block")
                    )
                )
            grad_op_desc, op_grad_to_var = core.get_grad_op_desc(
                op.desc, no_grad_dict[block.idx], []
            )

            # record the mapping between fwd and bwd
            if grad_op_id_to_fwd_op is not None:
                for op_desc in grad_op_desc:
                    grad_op_id_to_fwd_op[op_desc.original_id()] = op

            # Set device for grad_op according to forward Op
            if op.desc.has_attr(device_attr_name):
                op_device = op.desc.attr(device_attr_name)
                for op_desc in grad_op_desc:
                    op_desc._set_attr(device_attr_name, op_device)
            added_descs = _add_descs_to_block(
                grad_op_desc, local_block, grad_op_id_to_fwd_op
            )
            grad_op_descs.extend(added_descs)
            grad_to_var.update(op_grad_to_var)

    for i, segment in enumerate(recompute_segments[::-1]):
        gap_ops = ops[segment[1] : max_calculated_op_position]
        max_calculated_op_position = segment[0]
        for op in reversed(gap_ops):
            if op.has_attr("sub_block"):
                raise Exception(
                    "Recompute don't support ops with sub_block"
                    "invoke op: {}".format(
                        _pretty_op_desc_(op.desc, "with_sub_block")
                    )
                )
            grad_op_desc, op_grad_to_var = core.get_grad_op_desc(
                op.desc, no_grad_dict[block.idx], []
            )

            # record the mapping between fwd and bwd
            if grad_op_id_to_fwd_op is not None:
                for op_desc in grad_op_desc:
                    grad_op_id_to_fwd_op[op_desc.original_id()] = op

            # Set device for grad_op according to forward Op
            if op.desc.has_attr(device_attr_name):
                op_device = op.desc.attr(device_attr_name)
                for op_desc in grad_op_desc:
                    op_desc._set_attr(device_attr_name, op_device)
            added_descs = _add_descs_to_block(
                grad_op_desc, local_block, grad_op_id_to_fwd_op
            )
            grad_op_descs.extend(added_descs)
            grad_to_var.update(op_grad_to_var)

        ff_ops = ops[segment[0] : segment[1]]
        var_suffix = f".subprog_{i}"

        for op in ff_ops:
            if op.has_attr("sub_block"):
                raise Exception(
                    "Recompute don't support ops with sub_block"
                    "invoke op: {}".format(
                        _pretty_op_desc_(op.desc, "with_sub_block")
                    )
                )
            input_and_output_names = []
            input_and_output_names.extend(op.desc.input_arg_names())
            input_and_output_names.extend(op.desc.output_arg_names())
            for name in input_and_output_names:
                if block.var(name).persistable or name in checkpoints_name:
                    continue
                if name in vars_should_be_hold:
                    continue
                if name not in var_name_dict:
                    var_name_dict[name] = name + var_suffix

                    # we should create the rename var in subprog, otherwise its VarType will be BOOL
                    ref_var = block.program.global_block().var(name)
                    block.create_var(
                        name=var_name_dict[name],
                        shape=ref_var.shape,
                        dtype=ref_var.dtype,
                        type=ref_var.type,
                        persistable=ref_var.persistable,
                        stop_gradient=ref_var.stop_gradient,
                    )

        # 3.a. add ops in current recompute_segment as forward recomputation ops
        buffer_descs = _add_needed_descs_to_block(
            ff_ops, buffer_block, block, vars_in_memory, grad_op_id_to_fwd_op
        )
        added_descs = _add_descs_to_block(
            ff_ops, local_block, grad_op_id_to_fwd_op
        )

        # 3.b. rename all non-checkpoint variables in recomputation ops
        for key in var_name_dict:
            _rename_arg_(buffer_descs, key, var_name_dict[key])

        # added_descs should be in grad_op_descs because it is backward op desc
        grad_op_descs.extend(buffer_descs)

        # 3.c. add backward ops for all ops in current segment
        for op_desc in reversed(added_descs):
            grad_op_desc, op_grad_to_var = core.get_grad_op_desc(
                op_desc, no_grad_dict[block.idx], []
            )

            # record the mapping between fwd and bwd
            if grad_op_id_to_fwd_op is not None:
                for g_op_desc in grad_op_desc:
                    grad_op_id_to_fwd_op[g_op_desc.original_id()] = (
                        grad_op_id_to_fwd_op[op_desc.original_id()]
                    )

            # Set device for grad_op according to forward Op
            if op_desc.has_attr(device_attr_name):
                op_device = op_desc.attr(device_attr_name)
                for g_op_desc in grad_op_desc:
                    g_op_desc._set_attr(device_attr_name, op_device)

            for key in var_name_dict:
                _rename_arg_(grad_op_desc, key, var_name_dict[key])
            grad_op_descs.extend(grad_op_desc)
            grad_to_var.update(op_grad_to_var)

    # 3.d. add sum op for repetitive_outputs
    grad_op_descs = _addup_repetitive_outputs_(
        grad_op_descs, block.idx, grad_op_id_to_fwd_op=grad_op_id_to_fwd_op
    )
    # 4) remove no grad branch as it is in _remove_no_grad_branch_
    grad_op_descs = _remove_no_grad_branch_(
        grad_op_descs,
        no_grad_dict[block.idx],
        grad_op_id_to_fwd_op,
        target_vars,
    )
    added_descs = _add_descs_to_block(
        grad_op_descs, target_block, grad_op_id_to_fwd_op
    )
    return (
        program_stat,
        checkpoints_name,
        vars_should_be_hold,
        recompute_segments,
    )


def _get_sub_block_path(
    sub_block,
    sub_block_op_desc,
    no_grad_set,
    op_path_dict,
    sub_block_target_names=None,
):
    """
    Get output vars in subblock which will be assigned to parent block.
    It is used to find the grad path in subblock.

    Args:
        sub_block(Block): The sub-block in which to get op path.
        sub_block_op_desc: The op desc of the sub-block op such as 'while', 'conditional_block'.
        no_grad_set(set): The set of no grad var name. no_grad_set will be changed.
        op_path_dict(dict): op_path_dict will be changed.
            key(int) block index
            val(list) the op path of block(index)
        sub_block_target_names(set): Target var names of sub-block.
    Return:
        The forward op path of sub-block corresponding to backward op.
    """

    assert sub_block_op_desc.has_attr(
        "sub_block"
    ) and sub_block.idx == sub_block_op_desc._block_attr_id("sub_block")
    assert isinstance(sub_block_target_names, (set, type(None)))

    if sub_block_target_names is None:
        sub_block_target_names = sub_block_op_desc.output_arg_names

    # TODO(huihuangzheng): add support for recurrent op.
    if sub_block_op_desc.type in ["conditional_block", "while"]:
        # Step1: get the output vars in sub-block
        sub_outputs = [
            sub_block._var_recursive(var) for var in sub_block_target_names
        ]
        for var in sub_block_target_names:
            for op_desc in sub_block.ops:
                if var in op_desc.output_arg_names:
                    for name in op_desc.input_arg_names:
                        sub_outputs.append(sub_block._var_recursive(name))

        # Step2: find op path of sub-block
        is_while = sub_block_op_desc.type in ["while"]
        sub_block_op_path = _find_op_path_(
            sub_block, sub_outputs, [], no_grad_set, op_path_dict, is_while
        )
        return sub_block_op_path
    return sub_block.ops


def _is_grad_op_(op):
    op_maker = core.op_proto_and_checker_maker
    backward = core.op_proto_and_checker_maker.OpRole.Backward
    if op_maker.kOpRoleVarAttrName() in op.attr_names and int(
        op.all_attrs()[op_maker.kOpRoleAttrName()]
    ) == int(backward):
        return True
    return False


def _rename_grad_name_(name, grad_order):
    return 'grad/' * grad_order + name


def _topo_order_map(block, target_vars):
    """Analysis forward block and build a mapping from:
    OpDesc -> Int
    """
    get_defined_op = {}  # mapping from String -> OpDesc (defined op)
    for op in block.ops:
        for out_name in op.output_arg_names:
            get_defined_op[out_name] = op

    topo_order_map = {}  # mapping from OpDesc -> Topologic Order
    queue = [var.name for var in target_vars]
    visited = {var.name for var in target_vars}
    topo_order_counter = 0
    while len(queue) > 0:
        cur_var_name = queue.pop(0)
        if cur_var_name not in get_defined_op:
            continue
        cur_op = get_defined_op[cur_var_name]
        topo_order_map[cur_op] = topo_order_counter
        topo_order_counter += 1
        for inp in cur_op.input_arg_names:
            if inp in get_defined_op and inp not in visited:
                queue.append(inp)
                visited.add(inp)
    return topo_order_map


def _topo_bwd_order_map(topo_fwd_map, backward_op_map):
    topo_bwd_map = {}
    topo_fwd_map = {op.desc: order for op, order in topo_fwd_map.items()}
    for fwd_op, bwd_ops in backward_op_map.items():
        if fwd_op not in topo_fwd_map:
            continue
        for bwd_op in bwd_ops:
            topo_bwd_map[bwd_op] = topo_fwd_map[fwd_op]
    return topo_bwd_map


def _append_backward_ops_(
    block,
    ops,
    target_vars,
    target_block,
    no_grad_dict,
    grad_to_var,
    callbacks=None,
    input_grad_names_set=None,
    op_path_dict=None,
    distop_context=None,
    rename_var_map=None,
    grad_op_id_to_fwd_op=None,
):
    """
    Create all grad ops, and insert them into given block

    Args:
        block(Block): the block where forward ops are
        ops(Op): the forward operators whose backward ops need to be added
        target_vars(list[Tensor]): the loss vars we want to calculate gradient.
        target_block(Block): the block which is going to hold new generated grad ops
        no_grad_dict(dict):
            key(int)  block index
            val(set) a set of variable names. These variables have no gradient
        grad_to_var(dict)(output argument):
            key(str): grad variable name
            val(str): corresponding forward variable name
        callbacks(callable object): a callable object used to decorate new generated grad ops
        input_grad_names_set(set): this set is used to store the gradients' name which is
            generated by backward ops, and input_grad_names_set can help to prune the unnecessary
            backward ops.
        op_path_dict(dict): op_path_dict will be changed.
            key(int) block index
            val(list) the op path of block(index)
        rename_var_map(dict): used to associate target_grad var name with first grad_op input name.
            Only used in for high order gradient.
    """

    # Build the mapping between the forward op and backward op (Only for auto parallel)
    def update_distop_context(
        distop_context, op_grad_to_var, appending_grad_times
    ):
        distop_context.grad_var_to_var[appending_grad_times].update(
            op_grad_to_var
        )
        for op_desc in grad_op_desc:
            assert (
                op_desc.original_id() not in distop_context.grad_op_id_to_op_id
            )
            distop_context.grad_op_id_to_op_id[op_desc.original_id()] = (
                op.desc.original_id()
            )

    if callbacks is not None:
        assert isinstance(callbacks, (list, tuple))
        for cb in callbacks:
            if not callable(cb):
                raise ValueError("'callback' must be a callable object.")

    # grad_op_descs holds created grad_op, and will be appended to target_block
    grad_op_descs = []
    program = block.program
    get_backward_op_desc = {}  # for topo order map

    if rename_var_map is None:
        rename_var_map = {}
    assert isinstance(rename_var_map, dict)

    if core._is_bwd_prim_enabled():
        composite_block = program.clone().current_block()
        # Create output and infer shape for operators whose output haven't
        # been created.
        for op in composite_block.ops:
            for name in op.output_arg_names:
                if not (
                    composite_block.desc.has_var_recursive(name.encode())
                    or name == core.empty_var_name()
                ):
                    composite_block.create_var(name=name)
            op.desc.infer_var_type(composite_block.desc)
            op.desc.infer_shape(composite_block.desc)

    # add grad_op_desc by reversed ops
    for op in reversed(ops):
        grad_sub_block_list = []
        # If the op has its own sub-block, deal with the sub-block first
        if op.has_attr("sub_block"):
            sub_block = program.block(op._block_attr_id("sub_block"))
            grad_sub_block = program._create_block()
            grad_sub_block._set_forward_block_idx(sub_block.idx)
            # see following comments for why set None here.
            pre_input_grad_names_set = copy.copy(input_grad_names_set)
            input_grad_names_set = None
            sub_block_path = op_path_dict[op._block_attr_id("sub_block")]
            _append_backward_ops_(
                sub_block,
                sub_block_path,
                target_vars,
                grad_sub_block,
                no_grad_dict,
                grad_to_var,
                callbacks,
                input_grad_names_set,
                op_path_dict,
                grad_op_id_to_fwd_op=grad_op_id_to_fwd_op,
            )
            input_grad_names_set = pre_input_grad_names_set

            program._rollback()
            grad_sub_block_list.append(grad_sub_block.desc)
        # In primitive mode, raw phi GradOp will be split into multiple small
        # primitive operators, and the split rules are defined in c++ level,
        # see details: paddle/base/prim/api/manual/backward/composite_backward_api.h
        # It means that the output's shape and dtype of previous operators which
        # maybe used as the input of next operators must be known. Therefore,
        # we infer shape and dtype in a sandbox block(named composite_block) for
        # used in c++ level.
        # For example:
        #   forward:
        #       z = multiply(x, y) //maybe broadcast in kernel
        #   backward:
        #       x_grad_unreduce = z_grad * y // maybe unreduce
        #       reduced_axes = get_reduced_axes(x_grad.shape, x.shape) // need known shape
        #       x_grad = reduce_sum(x_grad_unreduce)
        grad_op_desc = []
        op_grad_to_var = {}
        if core._is_bwd_prim_enabled():

            def find_op_index(block_desc, cur_op_desc):
                for idx in range(block_desc.op_size()):
                    if cur_op_desc == block_desc.op(idx):
                        return idx
                return -1

            grad_op_desc, op_grad_to_var = core.get_grad_op_desc(
                composite_block.desc.op(find_op_index(block.desc, op.desc)),
                no_grad_dict[composite_block.idx],
                grad_sub_block_list,
            )
            for desc in grad_op_desc:
                infershape_for_composite(composite_block, desc)
        else:
            # Getting op's corresponding grad_op
            grad_op_desc, op_grad_to_var = core.get_grad_op_desc(
                op.desc, no_grad_dict[block.idx], grad_sub_block_list
            )

        # record the mapping between fwd and bwd
        get_backward_op_desc[op.desc] = grad_op_desc
        if grad_op_id_to_fwd_op is not None:
            for op_desc in grad_op_desc:
                grad_op_id_to_fwd_op[op_desc.original_id()] = op

        # Build the mapping between the forward op and backward op (Only for auto parallel)
        if distop_context is not None:
            update_distop_context(
                distop_context, op_grad_to_var, program._appending_grad_times
            )
        else:
            default_ctx = getattr(
                paddle.distributed.auto_parallel.static.dist_context,
                '_g_default_distributed_context',
                None,
            )
            if default_ctx is not None:
                distop_context = default_ctx.dist_op_context
                update_distop_context(
                    distop_context,
                    op_grad_to_var,
                    program._appending_grad_times,
                )

        # Set device for grad_op according to forward Op
        device_attr_name = core.op_proto_and_checker_maker.kOpDeviceAttrName()
        if op.desc.has_attr(device_attr_name):
            op_device = op.desc.attr(device_attr_name)
            for op_desc in grad_op_desc:
                op_desc._set_attr(device_attr_name, op_device)

        # Rename internal gradient variables in multiple backward
        # so that they have different names with previous backward.
        # For example:
        #  y = x * x, grad = base.gradients(base.gradients(y, x) + y * y, x)
        # In second-time backward, gradient variable names of partial
        # forward network (y * y) may be have same names with first-time
        # base.gradients(y, x).
        # So rename here before _addup_repetitive_outputs_.
        if program._appending_grad_times > 1:
            for op_desc in grad_op_desc:
                forward_op_inputs = op.desc.input_arg_names()
                for name in op_desc.input_arg_names():
                    if name in rename_var_map and name not in forward_op_inputs:
                        op_desc._rename_input(name, rename_var_map[name])
                for name in op_desc.output_arg_names():
                    if "@GRAD" not in name:
                        continue
                    if block.desc.find_var(name.encode("ascii")):
                        new_name = _rename_grad_name_(
                            name, program._appending_grad_times
                        )
                        op_desc._rename_output(name, new_name)
                        rename_var_map[name] = new_name

                        if name in op_grad_to_var:
                            # Build the mapping between the grad var name and var name (Only for auto parallel)
                            if distop_context is not None:
                                distop_context.grad_var_to_var[
                                    program._appending_grad_times
                                ][new_name] = op_grad_to_var[name]
                            op_grad_to_var[new_name] = op_grad_to_var[name]
                            op_grad_to_var.pop(name)

        # If input_grad_names_set is not None, extend grad_op_descs only when
        # any input grad in outputs of previous grad ops.
        # But this strategy is not suited for while op for some control flow,
        # for example, for while op, the grads maybe generated in next loop.
        if input_grad_names_set is not None:
            is_grad_name = (
                lambda name: name.find(core.grad_var_suffix()) != -1
                or name in input_grad_names_set
            )
            is_append_grad = False

            # NOTE: In primitive mode, the intermediate variable generated by
            # decompositing raw grad op are not satisfied the rule of 'XX@GRAD',
            # which will cause it be pruned according to current pruning logic.
            # For simplicity, we treat all primitive operators as one raw
            # operator, and keep the pruning logic consistent with currently
            # logic. The drawback of this solution is may lead to some primitive
            # operators are not pruned, which is needed to fixed.
            # FIXME: Optimize pruning logic from the perspective of whole graph.
            input_grad_names = []
            for op_desc in grad_op_desc:
                input_grad_names += [
                    name
                    for name in op_desc.input_arg_names()
                    if is_grad_name(name)
                ]

            # some code of gradient ops, like increment, are not very
            # standard, there is no @GRAD in these ops' inputs.
            if len(input_grad_names) == 0:
                is_append_grad = True
                continue

            if _some_in_set_(input_grad_names, input_grad_names_set):
                is_append_grad = True
                for op_desc in grad_op_desc:
                    grad_op_descs.append(op_desc)
                    for name in op_desc.output_arg_names():
                        input_grad_names_set.add(name)

            if is_append_grad:
                grad_to_var.update(op_grad_to_var)
        else:
            grad_op_descs.extend(grad_op_desc)
            grad_to_var.update(op_grad_to_var)

    # record mapping between grad var name and var name (Only for auto parallel)
    grad_var_to_var = None
    if distop_context is not None:
        grad_var_to_var = distop_context.grad_var_to_var[
            program._appending_grad_times
        ]
    # sum parameter's gradients' var given multiple var gradient
    if os.environ.get("FLAGS_program_topo_reorder", "False") in [
        'True',
        '1',
        'true',
    ]:
        topo_order = _topo_order_map(block, target_vars)
        topo_order_for_backward = _topo_bwd_order_map(
            topo_order, get_backward_op_desc
        )
    else:
        topo_order_for_backward = None
    grad_op_descs = _addup_repetitive_outputs_(
        grad_op_descs,
        block.idx,
        grad_var_to_var,
        grad_op_id_to_fwd_op=grad_op_id_to_fwd_op,
        topo_order_for_backward=topo_order_for_backward,
    )

    # if all outputs of the grad op are in no_grad_set, then just remove and fill zero
    # if all inputs of the grad op are in no_grad_set, just remove this op
    grad_op_descs = _remove_no_grad_branch_(
        grad_op_descs,
        no_grad_dict[block.idx],
        grad_op_id_to_fwd_op,
        target_vars,
    )

    # remove some backward ops
    # TODO(Jiabin): Support this in prime later, it will prune add_grad, fix this problem
    if not core._is_bwd_prim_enabled():
        not_need_ops = _find_not_need_ops(
            grad_op_descs, ops, input_grad_names_set
        )
        grad_op_descs = [
            op_desc for op_desc in grad_op_descs if op_desc not in not_need_ops
        ]
    else:
        logging.debug(
            "Running backward composite and disable find_not_need_ops"
        )

    # append op_desc in grad_op_descs to target_block
    op_role_attr_name = core.op_proto_and_checker_maker.kOpRoleAttrName()
    backward = core.op_proto_and_checker_maker.OpRole.Backward
    for op_desc in grad_op_descs:
        new_op_desc = target_block.desc.append_op()
        new_op_desc.copy_from(op_desc)
        new_op_desc._set_attr(op_role_attr_name, backward)
        grad_to_var["__current_op_desc__"] = new_op_desc
        if callbacks is not None:
            assert isinstance(callbacks, (list, tuple))
            for cb in callbacks:
                cb(block=target_block, context=grad_to_var)


def _is_grad_var_(var_name):
    return core.grad_var_suffix() in var_name


# Find the op who holds the sub_block as its "sub_block" attr
def _find_parent_op_(sub_block):
    sub_block_id = sub_block.idx

    if sub_block_id == 0:
        return None

    program = sub_block.program
    for block_id in range(program.num_blocks):
        block_desc = program.block(block_id).desc
        for op_idx in range(block_desc.op_size()):
            op = block_desc.op(op_idx)
            if (
                op.has_attr("sub_block")
                and op._block_attr_id("sub_block") == sub_block_id
            ):
                return op

    # NOTE(paddle-dev): When optimizer is added in conditional block,
    # sub_block may not be found.
    return None


def _append_backward_vars_(block, start_op_idx, grad_to_var, grad_info_map):
    """
    Create new variables required by backward pass.

    Args:
        block(Block): the block where new variables will be created
        start_op_idx(int): Only variables required by ops in block.ops[start_op_idx : ] will be created
        grad_to_var(dict):
            key(str): grad variable name
            val(str): corresponding forward variable name
            In most cases, this dict is generated by _append_backward_ops_()
        grad_info_map(dict)(output argument):
            key(str): forward variable name
            val(tuple): a tuple of (str, Block), str is the corresponding grad name, Block is the block containing grad variable
    """
    ops_to_remove = []
    '''
    NOTE(paddle-dev): while_grad op may hold some inputs which are not found
    in the parent/forward block, and they are also the outputs of while_grad
    op. These kinds of inputs are the recursive outputs inside while_grad op.
    They should be considered as "already created" when scanning the inner
    ops of while_grad ops.
    '''
    parent_op = _find_parent_op_(block)
    parent_op_vars = []
    if parent_op is not None:
        input_args = parent_op.input_arg_names()
        output_args = parent_op.output_arg_names()
        for in_arg in input_args:
            if in_arg in output_args:
                parent_op_vars.append(in_arg)

    for op_idx in range(start_op_idx, block.desc.op_size()):
        op_desc = block.desc.op(op_idx)
        if op_desc.has_attr("sub_block"):
            sub_block = block.program.block(op_desc._block_attr_id("sub_block"))
            _append_backward_vars_(sub_block, 0, grad_to_var, grad_info_map)

        grad_var_ins = [
            var for var in op_desc.input_arg_names() if _is_grad_var_(var)
        ]
        grad_var_outs = [
            var for var in op_desc.output_arg_names() if _is_grad_var_(var)
        ]

        inputs = [
            var
            for var in op_desc.input_arg_names()
            if var != core.empty_var_name()
        ]
        outputs = [
            var
            for var in op_desc.output_arg_names()
            if var != core.empty_var_name()
        ]

        # If the outputs of grad op is empty, just remove it
        if not outputs:
            ops_to_remove.append(op_idx)
            continue
        else:
            '''
            If the output is not empty and there is any grad input, find
            whether there is any existing input. If not, just remove it.
            '''
            if grad_var_ins:
                existing_grad_var_ins = [
                    var
                    for var in grad_var_ins
                    if block.desc.has_var_recursive(var.encode())
                    or var in parent_op_vars
                ]
                if not existing_grad_var_ins:
                    ops_to_remove.append(op_idx)
                    continue

        # sum may create invalid variable, here to deal with it.
        if op_desc.type() == 'sum':
            new_inputs = []
            for grad_var_name in op_desc.input_arg_names():
                if block.desc.has_var_recursive(grad_var_name.encode()):
                    # meet invalid sum variables, remove the invalid operand.
                    new_inputs.append(grad_var_name)
            assert (
                len(new_inputs) > 0
            ), "After remove invalid variables, sum op have no inputs."
            op_desc.set_input("X", new_inputs)

        new_vars = set()
        # create new gradient variables
        for grad_var_name in op_desc.output_arg_names():
            if (
                block.desc.has_var_recursive(grad_var_name.encode())
                or grad_var_name == core.empty_var_name()
            ):
                continue
            block.desc.var(grad_var_name.encode())
            new_vars.add(grad_var_name)
            if grad_var_name not in grad_to_var:
                continue
            grad_info_map[grad_to_var[grad_var_name]] = (grad_var_name, block)
        # infer_shape and infer_type
        op_desc.check_attrs()
        op_desc.infer_var_type(block.desc)
        op_desc.infer_shape(block.desc)

        for arg in op_desc.output_arg_names():
            if arg in new_vars:
                _infer_var_data_type_shape_(arg, block)

    for op_idx in reversed(ops_to_remove):
        block.desc._remove_op(op_idx, op_idx + 1)


def infershape_for_composite(block, grad_op_desc):
    # NOTE: why pruning the operator with empty output here ?
    # Some backward operator will output empty var, which will cause infer
    # shape error, such assign with input's stop_gradient=True
    if len(grad_op_desc.output_arg_names()) == 0:
        return

    # create output variable
    new_vars = set()
    for grad_var_name in grad_op_desc.output_arg_names():
        if not (
            block.desc.has_var_recursive(grad_var_name.encode())
            or grad_var_name == core.empty_var_name()
        ):
            # NOTE: stop_gradient will be set in append_op
            desc = block.desc.var(grad_var_name.encode())
            block.create_var(name=grad_var_name, desc=desc, type=desc.type())
            new_vars.add(grad_var_name)

    # NOTE For the primitive operator generated by decompositing phi grad kernel,
    # we Operator to reconstruct the op_desc for reusing some complex logic, such
    # as processing dispensable input, intermediate output, extra attrs, etc...
    if framework.OpProtoHolder.instance().has_op_proto(grad_op_desc.type()):
        op = block.append_op(
            type=grad_op_desc.type(),
            inputs={
                name: [block._find_var_recursive(arg) for arg in args]
                for name, args in grad_op_desc.inputs().items()
            },
            outputs={
                name: [block._find_var_recursive(arg) for arg in args]
                for name, args in grad_op_desc.outputs().items()
            },
            # NOTE Runtime attr will be ignore as the c++ GetRuntimeAttr
            # interface can't be exported to python. Please note the WARNING
            # message logged in RuntimeAttrs of composite_grad_desc_maker.h
            attrs=grad_op_desc.get_attr_map(),
        )
        op.desc._set_attr(
            core.op_proto_and_checker_maker.kOpRoleAttrName(),
            core.op_proto_and_checker_maker.OpRole.Backward,
        )
        grad_op_desc.copy_from(op.desc)
    # For the backward operator, we reuse the logic of _append_backward_var
    else:
        op_desc = block.desc.append_op()
        op_desc.copy_from(grad_op_desc)
        op_desc._set_attr(
            core.op_proto_and_checker_maker.kOpRoleAttrName(),
            core.op_proto_and_checker_maker.OpRole.Backward,
        )
        op_desc.check_attrs()
        op_desc.infer_var_type(block.desc)
        op_desc.infer_shape(block.desc)
        grad_op_desc.copy_from(op_desc)

    if not framework.OpProtoHolder.instance().has_op_proto(grad_op_desc.type()):
        # NOTE: Some raw base grad operators which hadn't been decomposed may not
        # implement InferVarType method, such as elementwise_xx_grad, and it will
        # cause the dtype or shape of corresponding cotangent incorrect. This
        # patch set the cotangent dtype and shape same with corresponding
        # forward variable. For primitive operators, we have ensure all
        # InferVarType method to be executed correctly in PR#52818, we skip
        # this patch for primitive operators.
        for arg in grad_op_desc.output_arg_names():
            if arg in new_vars:
                _infer_var_data_type_shape_(arg, block)


def _rename_grad_(
    block, start_op_idx, grad_to_var, target_grad_map, skip_rename_var_list
):
    var_map = copy.copy(target_grad_map)
    for op_idx in range(start_op_idx, block.desc.op_size()):
        op_desc = block.desc.op(op_idx)
        for name in op_desc.input_arg_names():
            if name in var_map:
                op_desc._rename_input(name, var_map[name])

        for name in op_desc.output_arg_names():
            if "@GRAD" not in name:
                continue
            if block.desc.find_var(name.encode("ascii")):
                if name in skip_rename_var_list:
                    continue
                new_name = unique_name.generate(name)
                op_desc._rename_output(name, new_name)
                var_map[name] = new_name

    for g, ng in var_map.items():
        if g in grad_to_var:
            grad_to_var[ng] = grad_to_var[g]
            grad_to_var.pop(g)


def _get_stop_gradients_(program):
    no_grad_dict = {}
    assert isinstance(program, framework.Program)
    for block in program.blocks:
        assert isinstance(block, framework.Block)
        block_no_grad_set = set()
        for var in list(block.vars.values()):
            assert isinstance(var, framework.Variable)
            if var.stop_gradient:
                block_no_grad_set.add(_append_grad_suffix_(var.name))
        no_grad_dict[block.idx] = block_no_grad_set
    return no_grad_dict


def _get_son_parent_block_idx_dict(program, current_block_idx):
    son_parent_block_idx_dict = collections.OrderedDict()
    while current_block_idx >= 0:
        parent_block_idx = program.block(current_block_idx).parent_idx
        son_parent_block_idx_dict[current_block_idx] = parent_block_idx
        current_block_idx = parent_block_idx

    return son_parent_block_idx_dict


def _get_no_grad_set_name(no_grad_set):
    no_grad_set_name = set()
    if no_grad_set is not None:
        if isinstance(no_grad_set, (set, list, tuple)):
            for i, no_grad_var in enumerate(no_grad_set):
                if isinstance(no_grad_var, framework.Variable):
                    no_grad_set_name.add(no_grad_var.name)
                elif isinstance(no_grad_var, str):
                    no_grad_set_name.add(no_grad_var)
                else:
                    raise TypeError(
                        f"The type of no_grad_set's member must be paddle.base.Variable or str, but received {type(no_grad_var)}."
                    )
        else:
            raise TypeError(
                f"The type of no_grad_set should be set or list or tuple, but received {type(no_grad_set)}"
            )
    return no_grad_set_name


def _get_no_grad_set_value(no_grad_set):
    no_grad_set_value = paddle.autograd.backward_utils.ValueSet()
    if no_grad_set is not None:
        if isinstance(no_grad_set, (set, list, tuple)):
            for i, no_grad_value in enumerate(no_grad_set):
                if isinstance(no_grad_value, paddle.pir.Value):
                    no_grad_set_value.add(no_grad_value)
                else:
                    raise TypeError(
                        f"The type of no_grad_set's member must be paddle.pir.Value, but received {type(no_grad_value)}."
                    )
        else:
            raise TypeError(
                f"The type of no_grad_set should be set or list or tuple, but received {type(no_grad_set)}"
            )
    return no_grad_set_value


@overload
def append_backward(
    loss: Tensor,
    parameter_list: Sequence[Tensor | str] | None = ...,
    no_grad_set: set[Tensor | str] | None = ...,
    callbacks: (
        Sequence[Callable[[Block, dict[str, Tensor | core.OpDesc]], None]]
        | None
    ) = ...,
    checkpoints: None = ...,
    distop_context: DistributedContext | None = ...,
) -> list[tuple[Tensor, Tensor]]: ...


@overload
def append_backward(
    loss: Tensor,
    parameter_list: Sequence[Tensor | str] | None = ...,
    no_grad_set: set[Tensor | str] | None = ...,
    callbacks: (
        Sequence[Callable[[Block, dict[str, Tensor | core.OpDesc]], None]]
        | None
    ) = ...,
    checkpoints: list[Tensor] = ...,
    distop_context: DistributedContext | None = ...,
) -> tuple[list[tuple[Tensor, Tensor]], list[str]]: ...


@framework.static_only
def append_backward(
    loss,
    parameter_list=None,
    no_grad_set=None,
    callbacks=None,
    checkpoints=None,
    distop_context=None,
):
    """
    :api_attr: Static Graph

    This function appends backward part to main_program.

    A complete neural network training is made up of forward and backward
    propagation. However, when we configure a network, we only need to
    specify its forward part. This function uses the chain rule to automatically
    generate the backward part according to the forward part.

    In most cases, users do not need to invoke this function manually.
    It will be automatically invoked by the optimizer's `minimize` function.

    Parameters:
        loss(Tensor): The loss Tensor of the network.
        parameter_list(list[Tensor|str]|tuple[Tensor|str], optional): List/Tuple of Parameters or Parameter.names
                                           that need to be updated by optimizers.
                                           If it is None, all parameters
                                           will be updated.
                                           Default: None.
        no_grad_set(set[Tensor|str], optional): Set of Tensors or Tensor.names in the :ref:`api_guide_Block_en` 0 whose gradients
                               should be ignored. All Tensors with
                               `stop_gradient=True` from all blocks will
                               be automatically added into this set.
                               If this parameter is not None, the Tensors or Tensor.names in this set will be added to the default set.
                               Default: None.
        callbacks(list[callable object]|tuple[callable object], optional): List/Tuple of callback functions.
                                               The callbacks are used for
                                               doing some custom jobs during
                                               backward part building. All
                                               callable objects in it will
                                               be invoked once each time a
                                               new gradient operator is added
                                               into the program. The callable
                                               object must have two input
                                               parameters: ``block`` and ``context`` .
                                               The ``block`` is the :ref:`api_guide_Block_en` which
                                               the new gradient operator will
                                               be added to. The ``context`` is a
                                               map, whose keys are gradient
                                               Tensor names and values are
                                               corresponding original :ref:`api_guide_tensor_en` .
                                               In addition to this, the ``context``
                                               has another special key-value pair:
                                               the key is string ``__current_op_desc__``
                                               and the value is the op_desc of the
                                               gradient operator who has just
                                               triggered the callable object.
                                               Default: None.

    Returns:
        list of tuple ( :ref:`api_guide_tensor_en` , :ref:`api_guide_tensor_en` ): Pairs of parameter and its corresponding gradients.
        The key is the parameter and the value is gradient Tensor.

    Raises:
        AssertionError: If ``loss`` is not an instance of Tensor.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> paddle.enable_static()

            >>> x = paddle.static.data(name='x', shape=[None, 13], dtype='int64')
            >>> y = paddle.static.data(name='y', shape=[None, 1], dtype='float32')
            >>> x_emb = paddle.static.nn.embedding(x, size=[100, 256])
            >>> y_predict = paddle.static.nn.fc(x=x_emb, size=1, activation=None, name='my_fc')
            >>> loss = F.square_error_cost(input=y_predict, label=y)
            >>> avg_loss = paddle.mean(loss)

            >>> # Get all weights in main_program, not include bias.
            >>> all_weights = [param for param in paddle.static.default_main_program().block(0).all_parameters() if 'w_' in param.name]
            >>> all_weights_name = [w.name for w in all_weights]

            >>> # return all param_grads needed to be updated if parameter_list set default None.
            >>> p_g_list1 = paddle.static.append_backward(loss=avg_loss)
            >>> # output: [(embedding_0.w_0, embedding_0.w_0@GRAD), (my_fc.w_0, my_fc.w_0@GRAD), (my_fc.b_0, my_fc.b_0@GRAD)]

            >>> # return the param_grads corresponding to parameter_list that can be list of param (Tensor).
            >>> p_g_list2 = paddle.static.append_backward(loss=avg_loss, parameter_list=all_weights)
            >>> # output: [(embedding_0.w_0, embedding_0.w_0@GRAD), (my_fc.w_0, my_fc.w_0@GRAD)]

            >>> # parameter_list can be list of param.name (str).
            >>> p_g_list3 = paddle.static.append_backward(loss=avg_loss, parameter_list=all_weights_name)
            >>> # output: [(embedding_0.w_0, embedding_0.w_0@GRAD), (my_fc.w_0, my_fc.w_0@GRAD)]

            >>> # no_grad_set can be set of Tensors that means grad will be cut off from these Tensors.
            >>> p_g_list4 = paddle.static.append_backward(loss=avg_loss, no_grad_set=set([x_emb]))
            >>> # output: [(my_fc.w_0, my_fc.w_0@GRAD), (my_fc.b_0, my_fc.b_0@GRAD)]

            >>> # no_grad_set can be set of Tensor.name when the Tensor is created inside layers and can't be specified explicitly.
            >>> p_g_list5 = paddle.static.append_backward(loss=avg_loss, no_grad_set=set(['my_fc.b_0']))
            >>> # output: [(embedding_0.w_0, embedding_0.w_0@GRAD), (my_fc.w_0, my_fc.w_0@GRAD)]

            >>> # return [] because all param_grads are filtered by no_grad_set.
            >>> p_g_list6 = paddle.static.append_backward(loss=avg_loss, parameter_list=all_weights, no_grad_set=set(all_weights))

    """
    if framework.in_pir_mode():
        return paddle.autograd.ir_backward.append_backward(
            loss, parameter_list, no_grad_set
        )

    grad_op_id_to_fwd_op = (
        {}
    )  # for cuda graph usage, recording the mapping between grad op original id to fwd op

    check_type(
        loss, 'loss', framework.Variable, 'paddle.static.append_backward'
    )

    if loss.op is None:
        # the loss is from a cloned program. Find loss op manually.
        _find_loss_op_(loss)

    loss.op._set_attr(
        core.op_proto_and_checker_maker.kOpRoleAttrName(),
        int(core.op_proto_and_checker_maker.OpRole.Forward)
        | int(core.op_proto_and_checker_maker.OpRole.Loss),
    )

    if callbacks is not None:
        check_type(
            callbacks,
            'callbacks',
            (list, tuple),
            'paddle.static.append_backward',
        )

    program = loss.block.program
    root_block = program.block(0)
    current_block_idx = program.current_block_idx
    current_block = program.block(current_block_idx)

    is_in_control_flow = current_block_idx != 0

    # Double grad is not supported in sub-block (control flow)
    if not is_in_control_flow:
        # _appending_grad_times used for double grad
        program._appending_grad_times += 1

    if no_grad_set is None:
        no_grad_set = set()
    else:
        no_grad_set = _get_no_grad_set_name(copy.copy(no_grad_set))
    no_grad_dict = _get_stop_gradients_(program)
    # no_grad_set only contains vars in block 0
    # Todo(liym27): support vars in sub block
    no_grad_dict[0].update(list(map(_append_grad_suffix_, no_grad_set)))

    # Currently it is only to support the optimizer.minimize
    # in a switch branch, which can append_backward in a sub_block.
    # Note: while_loop is in control flow, but it makes no sense to call optimizer in while.
    # Todo: report error when it is in while_loop
    if is_in_control_flow:
        # create grad block if in switch control flow.
        target_grad_block = program._create_block(
            parent_idx=current_block.parent_idx
        )
        target_grad_block._set_forward_block_idx(current_block_idx)
        # after _create_block, program.current_block changes
    else:
        target_grad_block = root_block

    son_parent_block_idx_dict = _get_son_parent_block_idx_dict(
        program, current_block_idx
    )

    block_fwd_op_num_dict = {}  # block_id: fwd_op_num
    for idx in son_parent_block_idx_dict:
        block_fwd_op_num_dict[idx] = program.block(idx).desc.op_size()

    grad_to_var = {}

    # pass the cuda_graph_attr to the fill_constant which generates the loss_grad
    op_desc = _create_loss_op_desc_(loss)
    grad_op_id_to_fwd_op[op_desc.original_id()] = loss.op
    target_grad_block.desc.append_op().copy_from(op_desc)

    for block_idx in son_parent_block_idx_dict:
        block = program.block(block_idx)

        block_no_grad_set = set(
            map(_strip_grad_suffix_, no_grad_dict[block_idx])
        )

        op_path_dict = {}
        op_path = _find_op_path_(
            block, [loss], [], block_no_grad_set, op_path_dict
        )

        no_grad_set = _find_no_grad_vars(
            block, op_path, [loss], block_no_grad_set
        )

        block_no_grad_set.update(no_grad_set)
        no_grad_dict[block_idx].update(
            list(map(_append_grad_suffix_, block_no_grad_set))
        )

        input_grad_names_set = None
        # For double backward, input_grad_names is used for filtering
        # some non-used gradients op(s).

        # TODO(liym27): need a better design.
        # not support double grad in control flow sub-block now.
        if not is_in_control_flow:
            if program._appending_grad_times > 1:
                input_grad_names_set = {_append_grad_suffix_(loss.name)}

        # TODO: support _append_backward_ops_with_checkpoints_ in
        #  sub-block (control flow)
        is_recompute = False
        if (
            checkpoints is not None
            and isinstance(checkpoints, list)
            and len(checkpoints) > 0
        ):
            is_recompute = True
            (
                program_stat,
                checkpoint_names,
                vars_should_be_hold,
                recompute_segments,
            ) = _append_backward_ops_with_checkpoints_(
                root_block,
                op_path,
                [loss],
                root_block,
                no_grad_dict,
                grad_to_var,
                checkpoints,
                grad_op_id_to_fwd_op,
            )
        else:
            _append_backward_ops_(
                block,  # the block where forward ops are in
                op_path,
                [loss],
                target_grad_block,
                no_grad_dict,
                grad_to_var,
                callbacks,
                input_grad_names_set=input_grad_names_set,
                op_path_dict=op_path_dict,
                distop_context=distop_context,
                grad_op_id_to_fwd_op=grad_op_id_to_fwd_op,
            )

    grad_info_map = {}

    # if in control flow, target_grad_block is a created new block which only contains grad ops,
    # so fwd_op_num is set to 0.
    fwd_op_num = (
        block_fwd_op_num_dict[current_block_idx]
        if not is_in_control_flow
        else 0
    )

    # Because append_backward may be called multiple times,
    # we need rename the internal gradient variables so that they have
    # different names.
    _rename_grad_(target_grad_block, fwd_op_num, grad_to_var, {}, [])

    _append_backward_vars_(
        target_grad_block, fwd_op_num, grad_to_var, grad_info_map
    )

    program.current_block_idx = current_block_idx
    program._sync_with_cpp()

    # for cuda graph, copy the cuda graph attr from forward op to backward op
    for op in target_grad_block.ops:
        if grad_op_id_to_fwd_op.get(op.desc.original_id(), None) is not None:
            fwd_op = grad_op_id_to_fwd_op[op.desc.original_id()]
            op._cuda_graph_attr = fwd_op._cuda_graph_attr

    if parameter_list is not None:
        check_type(
            parameter_list,
            'parameter_list',
            (list, tuple, set),
            'base.backward.append_backward',
        )
        parameters = []
        for i, param in enumerate(parameter_list):
            check_type(
                param,
                f'parameter_list[{i}]',
                (framework.Variable, str),
                'base.backward.append_backward',
            )
            if isinstance(param, framework.Variable):
                parameters.append(param.name)
            elif isinstance(param, str):
                parameters.append(param)
    else:
        params = program.global_block().all_parameters()
        parameters = [param.name for param in params if param.trainable]

    params_and_grads = []
    op_role_var_attr_name = core.op_proto_and_checker_maker.kOpRoleVarAttrName()
    for param in parameters:
        if param not in grad_info_map:
            continue
        grad_info = grad_info_map[param]
        grad_block = grad_info[1]
        if not grad_block.has_var(grad_info[0]):
            raise ValueError(
                f"grad block[{grad_info[1]}] did not have grad var {grad_info[0]}"
            )
        # Get the param var from the global block
        param_var = program.global_block().var(param)
        grad_var = grad_block.var(grad_info[0])
        if not is_in_control_flow:
            if loss.block.has_var(grad_info[0]):
                params_and_grads.append((param_var, grad_var))
            else:
                params_and_grads.append((param_var, None))
        else:
            params_and_grads.append((param_var, grad_var))

    for p, g in params_and_grads:
        if g is None:
            continue
        ops = (
            grad_block.ops if is_in_control_flow else program.global_block().ops
        )
        for op in reversed(ops):
            assert isinstance(op, framework.Operator)
            if g.name in op.output_arg_names:
                g.op = op
                break

        if g.op is None:
            raise ValueError("Unexpected branch")
        attr_val = [p.name, g.name]
        if g.op.has_attr(op_role_var_attr_name):
            attr_val.extend(g.op.attr(op_role_var_attr_name))
        g.op._set_attr(op_role_var_attr_name, attr_val)

    if is_recompute:
        return params_and_grads, checkpoint_names
    else:
        return params_and_grads


def _as_list(x):
    if x is None:
        return []
    return list(x) if isinstance(x, Sequence) else [x]


def _is_ancestor_block(ancestor_block, block):
    prog = block.program
    ancestor_idx = ancestor_block.idx
    parent_idx = block.parent_idx

    while parent_idx != -1:
        if parent_idx == ancestor_idx:
            return True
        parent_idx = prog.block(parent_idx).parent_idx

    return False


def _get_output_names(cur_block, targets):
    """
    In `cur_block`, get output names those linked to targets.
    NOTE:
    1. `targets` can be in `cur_block`;
    Usually, `targets` is in `cur_block`. However, considering control flow,
    2. `targets` may be in sub-block but `cur_block` is an ancestor of `targets[0].block`;
    3. `targets` may be in the block which is ancestor of `cur_block`.
    """

    block = targets[0].block if targets else cur_block
    current_output_names = {out.name for out in targets}

    # 1. If `targets` in cur_block or the ancestral block of `cur_block`
    if block.idx == cur_block.idx or _is_ancestor_block(block, cur_block):
        return current_output_names

    # 2. If `cur_block` is an ancestor of `targets[0].block`, run while loop
    prog = cur_block.program
    while block.idx != cur_block.idx:
        assert block.parent_idx != -1
        parent_block = prog.block(block.parent_idx)

        parent_block_output_names = set()
        for op in reversed(block.ops):
            if _some_in_set_(op.desc.output_arg_names(), current_output_names):
                for name in op.desc.input_arg_names():
                    current_output_names.add(name)
                    if not block.desc.find_var(
                        name.encode()
                    ) and parent_block.desc.find_var(name.encode()):
                        parent_block_output_names.add(name)

        block = parent_block
        current_output_names = parent_block_output_names

    return current_output_names


def _find_no_grad_vars(block, op_path, targets, no_grad_set):
    """
    Find the vars which is not used in the program, and
    those vars belong to no_grad_var.
    """
    output_names = _get_output_names(block, targets)
    no_grad_var = []
    for i, op in reversed(list(enumerate(op_path))):
        # If the op has sub_block, it is too complicated to find the correct no_grad_var.
        if not op.has_attr("sub_block"):
            for out_var in op.desc.output_arg_names():
                if (
                    out_var not in output_names
                    and out_var not in op.desc.input_arg_names()
                    and not block.vars[out_var].stop_gradient
                ):
                    no_grad_var.append(out_var)
        for name in op.desc.input_arg_names():
            if name not in no_grad_set:
                output_names.add(name)
    return set(no_grad_var)


def _find_op_path_(
    block, targets, inputs, no_grad_set, op_path_dict=None, is_while=False
):
    """
    It is used to find the grad path in `block`.

    Args:
        block(Block): The block in which to get op path.
        targets(list[Variable]): The target variables.
        inputs(list[Variable]): The input variables.
        no_grad_set(set): The set of no grad var name. no_grad_set will be changed.
        op_path_dict(dict): op_path_dict will be changed. op_path_dict will be changed.
            key(int) block index
            val(list) the op path of block(index)
        is_while(bool): Whether or not `block` is while block
    Return:
        The forward op path of block corresponding to backward op.
    """

    input_names = {inp.name for inp in inputs}
    output_names = _get_output_names(block, targets)
    if op_path_dict is None:
        op_path_dict = {}

    relevant_op_flags = [True] * len(block.ops)

    # All the inputs of the block are used if inputs is empty,
    if inputs:
        for i, op in enumerate(block.ops):
            if _some_in_set_(
                op.desc.input_arg_names(), input_names
            ) and not core.has_empty_grad_op_maker(op.type):
                for name in op.desc.output_arg_names():
                    if name not in no_grad_set:
                        input_names.add(name)
            else:
                relevant_op_flags[i] = False

    for i, op in reversed(list(enumerate(block.ops))):
        if op.has_attr("sub_block"):
            sub_block_id = op._block_attr_id("sub_block")
            sub_block = block.program.block(sub_block_id)
            sub_block_target_names = output_names & set(op.output_arg_names)
            sub_block_path = _get_sub_block_path(
                sub_block, op, set(), op_path_dict, sub_block_target_names
            )
            op_path_dict[sub_block_id] = sub_block_path

        if _some_in_set_(
            op.desc.output_arg_names(), output_names
        ) and not core.has_empty_grad_op_maker(op.type):
            for name in op.desc.input_arg_names():
                if name not in no_grad_set:
                    output_names.add(name)
        else:
            relevant_op_flags[i] = False

    if is_while:
        # If block is while block, dealing with op specifically again.
        # TODO(liym27): Consider special types of ops.
        for i, op in reversed(list(enumerate(block.ops))):
            if relevant_op_flags[i] is False and _some_in_set_(
                op.desc.output_arg_names(), output_names
            ):
                relevant_op_flags[i] = True
                if not core.has_empty_grad_op_maker(op.type):
                    for name in op.desc.input_arg_names():
                        if name not in no_grad_set:
                            output_names.add(name)

    op_path = [
        block.ops[i] for i in range(len(block.ops)) if relevant_op_flags[i]
    ]

    if inputs:
        for op in op_path:
            for name in op.desc.input_arg_names():
                if name not in input_names and block.vars[name].stop_gradient:
                    no_grad_set.add(name)

    return op_path


def calc_gradient_helper(
    targets, inputs, target_gradients=None, no_grad_set=None
):
    '''
    Calculate gradient and return grad_info_map
    '''
    targets = _as_list(targets)
    inputs = _as_list(inputs)
    target_gradients = _as_list(target_gradients)

    block = targets[0].block
    prog = block.program
    # increase appending gradients times
    prog._appending_grad_times += 1
    block_idx = block.idx

    if not target_gradients:
        target_gradients = [None] * len(targets)

    if len(targets) != len(target_gradients):
        raise ValueError(
            "Should have the same number of target_gradients as targets"
        )

    if no_grad_set is None:
        no_grad_set = set()
    else:
        no_grad_set = _get_no_grad_set_name(copy.copy(no_grad_set))
    no_grad_dict = _get_stop_gradients_(prog)
    no_grad_dict[0].update(list(map(_append_grad_suffix_, no_grad_set)))

    fwd_op_num = block.desc.op_size()

    input_grad_names_set = set()

    target_grad_map = {}
    rename_var_map = {}
    skip_rename_var_list = []
    grad_name_set = set()
    for i, grad in enumerate(target_gradients):
        target = targets[i]
        grad_name = _append_grad_suffix_(target.name)
        if grad is None:
            op_desc = _create_op_desc_(
                "fill_any_like",
                {"X": [target.name]},
                {"Out": [grad_name]},
                {
                    "value": 1.0,
                    "dtype": target.dtype,
                },
            )
            block.desc.append_op().copy_from(op_desc)
            block.program._sync_with_cpp()
            input_grad_names_set.add(grad_name)
            skip_rename_var_list.append(grad_name)
        else:
            if target.block.idx != block_idx or target.block.program != prog:
                raise ValueError("all targets must be in the same block")
            if target.shape != grad.shape:
                raise ValueError(
                    f"The shapes of target and grad are different: {target.name} {grad.name}"
                )
            target_grad_map[_append_grad_suffix_(target.name)] = grad.name
            input_grad_names_set.add(grad.name)
            rename_var_map[grad_name] = grad.name

        grad_name_set.add(grad_name)

    if core._is_bwd_prim_enabled():
        core._set_prim_target_grad_name(target_grad_map)
    # For double backward, input_grad_names is used for filter
    # some non-used gradients op. rename_var_map is used to
    # associate target_grad var name with first grad_op input name.
    if prog._appending_grad_times == 1:
        input_grad_names_set = None
        rename_var_map = {}

    for input in inputs:
        if input.block.program != prog:
            raise ValueError("input must be in the same program as targets")
    block_no_grad_set = set(map(_strip_grad_suffix_, no_grad_dict[0]))

    op_path_dict = {}
    op_path = _find_op_path_(
        block, targets, inputs, block_no_grad_set, op_path_dict
    )

    # only for composite to add grad_var of the last forward op
    # who has more than one output, but targets only has one,
    # so targets_gradients only add one grad_var,
    # eg: op1 -> op2 -> var1 / var2 targets = var1,
    # targets_gradients = var1_grad, need to add var2_grad here.
    tmp_targets = targets

    if core._is_bwd_prim_enabled():
        for op in reversed(block.ops):
            if op.type == "fill_any_like":
                continue
            # Some outputs of composite op are not needed and will be removed.
            # Thus, those vars should not be added with another op.
            keep_var_list = []
            if op.type in core.ops_contain_none.keys():
                values = core.ops_contain_none[op.type]
                if isinstance(values, list):
                    none_vars = values
                else:
                    none_vars = values(op)
                for none_var_name in none_vars:
                    keep_var_list.append(op.output(none_var_name)[0])

            for var_name in op.desc.output_arg_names():
                if keep_var_list and (var_name in keep_var_list):
                    continue
                grad_var_name = _append_grad_suffix_(var_name)
                if grad_var_name not in grad_name_set:
                    op_desc = _create_op_desc_(
                        "fill_any_like",
                        {"X": [var_name]},
                        {"Out": [grad_var_name]},
                        {'value': 0, 'dtype': targets[0].dtype},
                    )
                    block.desc.append_op().copy_from(op_desc)
                    tmp_targets.append(block.var(var_name))
            break
        block.program._sync_with_cpp()

    # find no grad var by op_path
    no_grad_set = _find_no_grad_vars(
        block, op_path, tmp_targets, block_no_grad_set
    )
    block_no_grad_set.update(no_grad_set)

    no_grad_dict[0].update(list(map(_append_grad_suffix_, block_no_grad_set)))
    grad_to_var = {}
    grad_info_map = {}
    _append_backward_ops_(
        block,
        op_path,
        targets,
        block,
        no_grad_dict,
        grad_to_var,
        input_grad_names_set=input_grad_names_set,
        op_path_dict=op_path_dict,
        rename_var_map=rename_var_map,
    )

    # Because calc_gradient may be called multiple times,
    # we need rename the internal gradient variables so that they have
    # different names.
    _rename_grad_(
        block, fwd_op_num, grad_to_var, target_grad_map, skip_rename_var_list
    )

    _append_backward_vars_(block, fwd_op_num, grad_to_var, grad_info_map)
    prog._sync_with_cpp()
    return grad_info_map


def _get_grad_vars(grad_info_map, inputs):
    inputs = _as_list(inputs)
    grad_vars = []
    for input_var in inputs:
        if input_var.name not in grad_info_map:
            grad_vars.append(None)
        else:
            grad_info = grad_info_map[input_var.name]
            grad_block = grad_info[1]
            grad_var = grad_block.var(grad_info[0])
            grad_vars.append(grad_var)
    return grad_vars


def calc_gradient(targets, inputs, target_gradients=None, no_grad_set=None):
    """
    Backpropagate the gradients of targets to inputs.

    Args:
        targets(Tensor|list[Tensor]|tuple[Tensor]): The target Tensors
        inputs(Tensor|list[Tensor]|tuple[Tensor]): The input Tensors
        target_gradients (Tensor|list[Tensor]|tuple[Tensor], optional): The gradient Tensors
            of targets which has the same shape with targets, If None, ones will
            be created for them.
        no_grad_set(set[Tensor|str], optional): Set of Tensors or Tensor.names in the :ref:`api_guide_Block_en` 0 whose gradients
                               should be ignored. All Tensors with
                               `stop_gradient=True` from all blocks will
                               be automatically added into this set.
                               If this parameter is not None, the Tensors or Tensor.names in this set will be added to the default set.
                               Default: None.

    Return:
        (list[Tensor]): A list of gradients for inputs
        If an input does not affect targets, the corresponding gradient Tensor
        will be None
    """
    if framework.in_pir_mode():
        return paddle.autograd.ir_backward.calc_gradient(
            targets, inputs, target_gradients, no_grad_set
        )

    # NOTE: If you want to modify the logic of calc_gradient, please modify
    # it inside the calc_gradient_helper and _get_grad_vars functions
    # to ensure the correctness of dy2st mode.
    grad_info_map = calc_gradient_helper(
        targets,
        inputs,
        target_gradients=target_gradients,
        no_grad_set=no_grad_set,
    )

    grad_vars = _get_grad_vars(grad_info_map, inputs)

    if len(grad_vars) == 1:
        return grad_vars[0]
    else:
        return grad_vars


@framework.static_only
def gradients(
    targets: Tensor | Sequence[Tensor],
    inputs: Tensor | Sequence[Tensor],
    target_gradients: Tensor | Sequence[Tensor] | None = None,
    no_grad_set: set[Tensor | str] | None = None,
) -> list[Tensor]:
    """

    Backpropagate the gradients of targets to inputs.

    Args:
        targets (Tensor|list[Tensor]|tuple[Tensor]): The target Tensors.
        inputs (Tensor|list[Tensor]|tuple[Tensor]): The input Tensors.
        target_gradients (Tensor|list[Tensor]|tuple[Tensor]|None, optional): The gradient Tensor
            of targets which has the same shape with targets, If None, ones will
            be created for them.
        no_grad_set (set[Tensor|str]|None, optional): Set of Tensors or Tensor.names in the :ref:`api_guide_Block_en` 0 whose gradients
            should be ignored. All Tensors with ``stop_gradient=True`` from all blocks will
            be automatically added into this set. If this parameter is not None, the Tensors or Tensor.names
            in this set will be added to the default set. Default: None.

    Return:
        (list[Tensor]): A list of gradients for inputs
        If an input does not affect targets, the corresponding gradient Tensor
        will be None.

    Examples:

        .. code-block:: python

            >>> # doctest: +SKIP("This has diff in xdoctest env")
            >>> import paddle
            >>> import paddle.nn.functional as F

            >>> paddle.enable_static()

            >>> x = paddle.static.data(name='x', shape=[None, 2, 8, 8], dtype='float32')
            >>> x.stop_gradient=False
            >>> y = paddle.static.nn.conv2d(x, 4, 1, bias_attr=False)
            >>> y = F.relu(y)
            >>> z = paddle.static.gradients([y], x)
            >>> print(z)
            [var x@GRAD : DENSE_TENSOR.shape(-1, 2, 8, 8).dtype(float32).stop_gradient(False)]
    """
    if framework.in_pir_mode():
        check_type(
            targets,
            'targets',
            (paddle.pir.Value, list, tuple),
            'paddle.autograd.ir_backward.grad',
        )
        check_type(
            inputs,
            'inputs',
            (paddle.pir.Value, list, tuple),
            'paddle.autograd.ir_backward.grad',
        )
        check_type(
            target_gradients,
            'target_gradients',
            (paddle.pir.Value, list, tuple, type(None)),
            'paddle.autograd.ir_backward.grad',
        )

        check_type(
            no_grad_set,
            'no_grad_set',
            (
                paddle.pir.Value,
                list,
                tuple,
                set,
                type(None),
            ),
            'paddle.autograd.ir_backward.grad',
        )
        targets = _as_list(targets)
        inputs = _as_list(inputs)
        target_gradients = _as_list(target_gradients)

        from paddle.autograd.backward_utils import ValueSet
        from paddle.autograd.ir_backward import (
            calc_gradient as pir_calc_gradient,
        )

        if no_grad_set is None:
            no_grad_set = ValueSet()
        else:
            no_grad_set = ValueSet(no_grad_set)

        input_grad = pir_calc_gradient(
            targets, inputs, target_gradients, no_grad_set
        )
        return input_grad

    check_type(
        targets,
        'targets',
        (framework.Variable, list, tuple),
        'paddle.static.gradients',
    )
    check_type(
        inputs,
        'inputs',
        (framework.Variable, list, tuple),
        'paddle.static.gradients',
    )
    check_type(
        target_gradients,
        'target_gradients',
        (framework.Variable, list, tuple, type(None)),
        'paddle.static.gradients',
    )
    outs = calc_gradient(targets, inputs, target_gradients, no_grad_set)
    return _as_list(outs)


@framework.static_only
def gradients_with_optimizer(program, optimizer, inputs=None, outputs=None):
    """
    :api_attr: Static Graph

    Backpropagate the gradients of the program and apply the gradients with the given optimizer.

    Args:
        program (Program): The input program.
        optimizer (Optimizer): The optimizer to apply the gradients.
        inputs (Tensor|list[Tensor]|tuple[Tensor], optional): The input Tensors.
            If None, the inputs will be created from the input variables in the given program. Default:None.
        outputs (Tensor|list[Tensor]|tuple[Tensor], optional): The output Tensors.
            If None, the outputs will be created from the output variables in the given program. Default: None.

    Return:
        tuple: tuple (optimize_ops, params_grads), A list of operators appended
            by gradients_with_optimizer and a list of (param, grad) variable pairs, param is
            ``Parameter``, grad is the gradient value corresponding to the parameter.
            The returned tuple can be passed to ``fetch_list`` in ``Executor.run()`` to
            indicate program pruning. If so, the program will be pruned by ``feed`` and
            ``fetch_list`` before run, see details in ``Executor``.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.static as static

            >>> paddle.enable_static()

            >>> img = static.data(name='image', shape=[None, 784])
            >>> pred = static.nn.fc(x=img, size=10, activation='relu')
            >>> loss = paddle.mean(pred)
            >>> opt = paddle.optimizer.SGD(learning_rate=1e-3)
            >>> opt_ops, pram_grads = paddle.base.backward.gradients_with_optimizer(static.default_main_program(), opt)
            >>> print(opt_ops)
            [{ParamOut=['fc_0.b_0']} = sgd(inputs={Grad=['fc_0.b_0@GRAD'],
            LearningRate=['learning_rate_0'],
            MasterParam=[],
            ...
            with_quant_attr = False)]

    """
    check_type(
        program,
        'program',
        paddle.base.Program,
        'paddle.static.gradients_with_optimizer',
    )
    check_type(
        optimizer,
        'optimizer',
        paddle.optimizer.Optimizer,
        'paddle.static.gradients_with_optimizer',
    )

    if inputs is None or outputs is None:
        in_set = set()
        out_set = set()
        for block in program.blocks:
            for op in block.ops:
                for name in op.input_arg_names:
                    in_set.add(block.vars[name])
                for name in op.output_arg_names:
                    out_set.add(block.vars[name])
        if inputs is None:
            inputs = list(in_set.difference(out_set))
        if outputs is None:
            outputs = list(out_set.difference(in_set))

    grads = gradients(outputs, inputs)

    with program_guard(program, None):
        pram_grads = [
            (pram, grad)
            for pram, grad in zip(inputs, grads)
            if isinstance(pram, paddle.base.framework.Parameter)
            and grad is not None
        ]

        optimize_ops = optimizer.apply_gradients(pram_grads)

    return optimize_ops, pram_grads
