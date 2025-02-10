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

import numpy as np

import paddle
from paddle.framework import core
from paddle.utils import unique_name

from .pass_base import PassBase, PassType, register_pass


def find_adjacent_match_sequences(
    iterable, filter_func, adjacent_filter_func=None
):
    n = len(iterable)
    match_sequences = []
    if adjacent_filter_func is None:
        adjacent_filter_func = lambda ref_op, new_op: True
    i = 0
    while True:
        while i < n and not filter_func(iterable[i]):
            i += 1
        j = i + 1
        while (
            j < n
            and filter_func(iterable[j])
            and adjacent_filter_func(iterable[i], iterable[j])
        ):
            j += 1
        if i < n and j <= n:
            match_sequences.append((i, j))
        i = j + 1
        if i >= n:
            break
    return match_sequences


def insert_fuse_all_reduce_ops(
    block, reversed_op_indices, input_var_names, output_var_names, dtype, attrs
):
    fused_var = block.create_var(
        name=unique_name.generate(f"FusedOutput_{input_var_names[0]}"),
        dtype=dtype,
    )

    # FIXME(zengjinle): here we assume that we use
    # c_sync_calc_stream/c_sync_comm_stream to do sync.
    # But someone may use c_wait_compute/c_wait_comm instead.
    if not attrs["use_calc_stream"]:
        ring_id = attrs["ring_id"]
        new_op_indices = list(reversed_op_indices)

        for i, op_idx in enumerate(reversed_op_indices):
            prev_op_idx = op_idx - 1
            while (
                prev_op_idx >= 0
                and block.ops[prev_op_idx].type == "c_sync_calc_stream"
            ):
                new_op_indices.append(prev_op_idx)
                prev_op_idx -= 1

            if i > 0:
                next_op_idx = op_idx + 1
                n = len(block.ops)
                while (
                    next_op_idx < n
                    and block.ops[next_op_idx].type == "c_sync_comm_stream"
                ):
                    assert block.ops[next_op_idx].attr("ring_id") == ring_id
                    new_op_indices.append(next_op_idx)

        new_op_indices = list(set(new_op_indices))
        new_op_indices.sort(reverse=True)
        reversed_op_indices = new_op_indices

    insert_idx = reversed_op_indices[0] + 1
    op_role_key = core.op_proto_and_checker_maker.kOpRoleAttrName()

    concated_shapes = []
    concated_ranks = []
    for var_name in output_var_names:
        shape = block._find_var_recursive(var_name).shape
        concated_shapes.extend(shape)
        concated_ranks.append(len(shape))

    coalesce_tensor_op_kwargs = {
        "type": "coalesce_tensor",
        "inputs": {
            "Input": input_var_names,
        },
        "outputs": {
            "Output": output_var_names,
            "FusedOutput": fused_var,
        },
        "attrs": {
            "use_align": True,
            "dtype": dtype,
            "concated_shapes": concated_shapes,
            "concated_ranks": concated_ranks,
            op_role_key: attrs[op_role_key],
        },
    }

    if not attrs["use_calc_stream"]:
        block._insert_op_without_sync(
            insert_idx,
            type="c_sync_calc_stream",
            inputs={"X": fused_var},
            outputs={"Out": fused_var, op_role_key: attrs[op_role_key]},
        )
        insert_idx += 1

    # all_reduce sum should insert
    attrs["reduce_type"] = paddle.distributed.ReduceOp.SUM
    block._insert_op_without_sync(
        insert_idx,
        type="all_reduce",
        inputs={"x": fused_var},
        outputs={"out": fused_var},
        attrs=attrs,
    )

    for op_idx in reversed_op_indices:
        block._remove_op(op_idx)

    return coalesce_tensor_op_kwargs


def has_same_attrs(op1, op2, attr_names):
    for attr_name in attr_names:
        if op1.attr(attr_name) != op2.attr(attr_name):
            return False
    return True


def filter_all_collective_op_indices(block):
    # NOTE: should add more collective ops
    all_collective_ops = {
        "c_allreduce_sum",
        "c_allreduce_prod",
        "c_allreduce_max",
        "c_allreduce_min",
        "c_allgather",
        "c_broadcast",
        "broadcast",
        "all_gather",
        "all_reduce",
    }

    match_op_indices = []
    for i, op in enumerate(block.ops):
        if op.type in all_collective_ops:
            match_op_indices.append(i)
    return match_op_indices


def find_all_fuse_all_reduce_groups(block):
    collective_op_indices = filter_all_collective_op_indices(block)
    collective_ops = [block.ops[i] for i in collective_op_indices]

    def is_valid_allreduce_op(op):
        if op.type != "c_allreduce_sum" or op.attr("use_model_parallel"):
            return False
        in_var_name = op.input("X")[0]
        out_var_name = op.output("Out")[0]
        if in_var_name != out_var_name:
            return False
        in_var = block._find_var_recursive(in_var_name)
        assert in_var is not None
        if in_var.type != core.VarDesc.VarType.DENSE_TENSOR:
            return False
        shape = in_var.shape
        if any(s <= 0 for s in shape):
            return False
        return True

    same_attr_names = [
        "ring_id",
        "use_calc_stream",
        core.op_proto_and_checker_maker.kOpRoleAttrName(),
        core.op_proto_and_checker_maker.kOpDeviceAttrName(),
    ]

    def is_same_adjacent_op(ref_op, new_op):
        if not has_same_attrs(ref_op, new_op, same_attr_names):
            return False
        ref_op_in_var = block._find_var_recursive(ref_op.input("X")[0])
        new_op_in_var = block._find_var_recursive(new_op.input("X")[0])
        if ref_op_in_var.dtype != new_op_in_var.dtype:
            return False
        return True

    match_seqs = find_adjacent_match_sequences(
        collective_ops, is_valid_allreduce_op, is_same_adjacent_op
    )
    new_match_seqs = []
    for i, j in match_seqs:
        new_match_seqs.append([collective_op_indices[k] for k in range(i, j)])
    return new_match_seqs


def split_fuse_all_reduce_groups_by_deps(block, groups, op_deps):
    new_groups = []

    def insert_new_group(op_indices, start_idx, end_idx):
        if end_idx - start_idx > 1:
            new_groups.append(op_indices[start_idx:end_idx])

    for op_indices in groups:
        n = len(op_indices)
        assert n > 0
        if n == 1:
            continue

        start_idx = 0
        k = start_idx + 1
        while k < n:
            found_group = False
            for prev_idx in range(start_idx, k):
                dep = op_deps[op_indices[prev_idx]][op_indices[k]]
                if dep == core.Node.Dep.NoDep:
                    continue
                # [start_idx, k) is valid groups
                insert_new_group(op_indices, start_idx, k)
                start_idx = k
                break
            k += 1

        insert_new_group(op_indices, start_idx, k)

    return new_groups


def insert_coalesce_tensor_ops(block, coalesce_ops_kwargs):
    if not coalesce_ops_kwargs:
        return

    var_infos = {}
    for idx, op in enumerate(block.ops):
        for var in op.input_arg_names:
            if var not in var_infos:
                var_infos[var] = [idx, True]

        for var in op.output_arg_names:
            if var not in var_infos:
                var_infos[var] = [idx, False]

    n = len(block.ops)
    insert_idx_and_kwargs = []
    for group_idx, kwargs in enumerate(coalesce_ops_kwargs):
        all_vars = kwargs["inputs"]["Input"] + kwargs["outputs"]["Output"]
        min_op_idx = n
        copy_data = False
        for var in all_vars:
            if var not in var_infos:
                copy_data = True
                min_idx = 0
                break
            op_idx, is_input = var_infos[var]
            if is_input:
                copy_data = True
            min_op_idx = min(min_op_idx, op_idx)
        kwargs["attrs"]["copy_data"] = copy_data
        insert_idx_and_kwargs.append((min_op_idx, kwargs))

    insert_idx_and_kwargs.sort(key=lambda element: element[0], reverse=True)
    for idx, kwargs in insert_idx_and_kwargs:
        block._insert_op_without_sync(idx, **kwargs)


def insert_fuse_all_reduce_by_memory_size(block, groups, max_memory_size):
    op_role_key = core.op_proto_and_checker_maker.kOpRoleAttrName()
    op_role_var_key = core.op_proto_and_checker_maker.kOpRoleVarAttrName()
    op_device_key = core.op_proto_and_checker_maker.kOpDeviceAttrName()
    coalesce_ops_kwargs = []
    for group in reversed(groups):
        first_op = block.ops[group[0]]
        ring_id = first_op.attr("ring_id")
        use_calc_stream = first_op.attr("use_calc_stream")
        use_model_parallel = first_op.attr("use_model_parallel")
        op_role = first_op.attr(op_role_key)
        op_device = first_op.attr(op_device_key)

        attrs = {
            "ring_id": ring_id,
            "use_calc_stream": use_calc_stream,
            "use_model_parallel": use_model_parallel,
            op_role_key: op_role,
            op_device_key: op_device,
        }
        dtype = block._find_var_recursive(first_op.input("X")[0]).dtype
        sizeof = core.size_of_dtype(dtype)

        cur_mem_size = 0
        op_role_vars = []
        recorded_op_indices = []
        in_var_names = []
        out_var_names = []
        for op_idx in reversed(group):
            op = block.ops[op_idx]
            in_var_name = op.input("X")[0]
            out_var_name = op.output("Out")[0]
            in_var = block._find_var_recursive(in_var_name)
            mem_size = int(np.prod(in_var.shape)) * sizeof
            if cur_mem_size + mem_size > max_memory_size:
                if len(recorded_op_indices) > 1:
                    attrs[op_role_var_key] = op_role_vars
                    coalesce_op_kwargs = insert_fuse_all_reduce_ops(
                        block,
                        recorded_op_indices,
                        in_var_names,
                        out_var_names,
                        dtype,
                        attrs,
                    )
                    coalesce_ops_kwargs.append(coalesce_op_kwargs)

                cur_mem_size = 0
                op_role_vars = []
                recorded_op_indices = []
                in_var_names = []
                out_var_names = []

            cur_mem_size += mem_size
            recorded_op_indices.append(op_idx)
            in_var_names.append(in_var_name)
            out_var_names.append(out_var_name)
            if op.has_attr(op_role_var_key):
                op_role_vars.extend(op.attr(op_role_var_key))

        if len(recorded_op_indices) > 1:
            attrs[op_role_var_key] = op_role_vars
            coalesce_op_kwargs = insert_fuse_all_reduce_ops(
                block,
                recorded_op_indices,
                in_var_names,
                out_var_names,
                dtype,
                attrs,
            )
            coalesce_ops_kwargs.append(coalesce_op_kwargs)
    block._sync_with_cpp()
    insert_coalesce_tensor_ops(block, coalesce_ops_kwargs)


@register_pass("fuse_all_reduce")
class FuseAllReducePass(PassBase):
    def __init__(self):
        super().__init__()
        self.set_attr("max_memory_size", -1)

    def _check_self(self):
        max_memory_size = self.get_attr("max_memory_size")
        return max_memory_size > 0

    def _check_conflict(self, other_pass):
        return True

    def _type(self):
        return PassType.COMM_OPT

    # NOTE: why FuseAllReducePass can override apply_single_impl instead of
    # apply_impl? AllReduce is a collective operation, so the program of each
    # rank inside the same communication group should have the same
    # all_reduce sum operations. Therefore, FuseAllReducePass can override
    # apply_single_impl directly.
    def _apply_single_impl(self, main_program, startup_program, context):
        max_memory_size = self.get_attr("max_memory_size")
        op_deps = main_program.desc.get_op_deps()
        num_blocks = main_program.num_blocks
        for i in range(num_blocks):
            block = main_program.block(i)
            groups = find_all_fuse_all_reduce_groups(block)
            groups = split_fuse_all_reduce_groups_by_deps(
                block, groups, op_deps[i]
            )
            insert_fuse_all_reduce_by_memory_size(
                block, groups, max_memory_size
            )
        main_program._sync_with_cpp()
