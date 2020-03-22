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

from __future__ import print_function

import paddle.fluid.core as core
import paddle.fluid.framework as framework
from paddle.fluid.incubate.fleet.parameter_server.details.program_utils import delete_ops

OP_NAME_SCOPE = "op_namescope"
CLIP_OP_NAME_SCOPE = "@CLIP"
OP_ROLE_VAR_ATTR_NAME = core.op_proto_and_checker_maker.kOpRoleVarAttrName()
RPC_OP_ROLE_ATTR_NAME = core.op_proto_and_checker_maker.kOpRoleAttrName()
RPC_OP_ROLE_ATTR_VALUE = core.op_proto_and_checker_maker.OpRole.RPC
op_role_attr_name = core.op_proto_and_checker_maker.kOpRoleAttrName()


class DistributedMode:
    SYNC = 0
    ASYNC = 1
    HALF_ASYNC = 2
    GEO = 3


def _is_opt_role_op(op):
    # NOTE: depend on oprole to find out whether this op is for
    # optimize
    op_maker = core.op_proto_and_checker_maker
    optimize_role = core.op_proto_and_checker_maker.OpRole.Optimize
    if op_maker.kOpRoleAttrName() in op.attr_names and \
            int(op.all_attrs()[op_maker.kOpRoleAttrName()]) == int(optimize_role):
        return True
    return False


def delete_optimizer_pass(program):
    def _get_optimize_ops(_program):
        block = _program.global_block()
        opt_ops = []
        for op in block.ops:
            if _is_opt_role_op(op):
                # delete clip op from opt_ops when run in Parameter Server mode
                if OP_NAME_SCOPE in op.all_attrs() \
                        and CLIP_OP_NAME_SCOPE in op.attr(OP_NAME_SCOPE):
                    op._set_attr(
                        "op_role",
                        int(core.op_proto_and_checker_maker.OpRole.Backward))
                    continue
                opt_ops.append(op)
        return opt_ops

    def _delete_optimizer_op_and_vars(_program, optimize_ops):
        optimize_vars = []
        optimize_op_role_vars = []
        optimize_need_delete_vars = []

        for op in optimize_ops:
            optimize_vars.extend(op.input_arg_names)
            optimize_op_role_vars.extend(op.attr("op_role_var"))

        optimize_vars = list(set(optimize_vars))
        optimize_op_role_vars = list(set(optimize_op_role_vars))

        for var in optimize_vars:
            if var not in optimize_op_role_vars:
                optimize_need_delete_vars.append(var)
        need_delete_optimize_vars = list(set(optimize_need_delete_vars))

        delete_ops(_program, optimize_ops)
        for var in need_delete_optimize_vars:
            if _program.global_block().has_var(var):
                _program.global_block()._remove_var(var)

    optimizer_ops = _get_optimize_ops(program)
    _delete_optimizer_op_and_vars(program, optimizer_ops)

    return program


def distributed_ops_pass(program, trainer_id, pserver_endpoints):
    def _get_pull_sparse_ops(_program):
        pull_sparse_ops = {}
        op_types = {"lookup_table": "W"}
        for op in _program.global_block().ops:
            if op.type in op_types.keys() \
                    and op.attr('remote_prefetch') is True:
                param_name = op.input(op.input_names[op_types[op.type]])[0]

                ops = pull_sparse_ops.get(param_name, [])
                ops.append(op)
                pull_sparse_ops[param_name] = ops
        return pull_sparse_ops

    def _pull_sparse_fuse(_program, pull_sparse_ops):
        for param, ops in pull_sparse_ops.items():
            all_ops = program.global_block().ops
            op_idxs = [all_ops.index(op) for op in ops]
            inputs = [
                program.global_block().vars[op.input("Ids")[0]] for op in ops
            ]
            w = program.global_block().vars[ops[0].input("W")[0]]
            padding_idx = ops[0].attr("padding_idx")
            outputs = [
                program.global_block().vars[op.output("Out")[0]] for op in ops
            ]

            for idx in op_idxs[::-1]:
                program.global_block()._remove_op(idx)

            inputs_idxs = [-1] * len(inputs)
            outputs_idxs = [-1] * len(outputs)

            for idx, op in enumerate(program.global_block().ops):
                for i in range(0, len(op.output_names)):
                    outs = op.output(op.output_names[i])
                    for in_id, in_var in enumerate(inputs):
                        if in_var.name in outs:
                            inputs_idxs[in_id] = idx
                for i in range(0, len(op.input_names)):
                    ins = op.input(op.input_names[i])
                    for out_id, out_var in enumerate(outputs):
                        if out_var.name in ins:
                            outputs_idxs[out_id] = idx

            if min(outputs_idxs) - max(inputs_idxs) >= 1:
                distributed_idx = max(inputs_idxs) + 1

                program.global_block()._insert_op(
                    index=distributed_idx,
                    type="distributed_lookup_table",
                    inputs={"Ids": inputs,
                            'W': w},
                    outputs={"Outputs": outputs},
                    attrs={
                        "endpoints": pserver_endpoints,
                        "padding_idx": padding_idx,
                        "trainer_id": trainer_id
                    })
            else:
                raise ValueError(
                    "something wrong with Fleet, submit a issue is recommended")

    pull_sparse_ops = _get_pull_sparse_ops(program)
    _pull_sparse_fuse(program, pull_sparse_ops)
    return program


def append_send_ops_pass(program, origin_program, mode, trainer_id,
                         pserver_endpoints):
    def _get_params_grads(sparse_varnames):
        block = origin_program.global_block()

        dense_param_grads = []
        sparse_param_grads = []

        optimize_params = set()
        origin_var_dict = origin_program.global_block().vars
        role_id = int(core.op_proto_and_checker_maker.OpRole.Backward)
        for op in block.ops:
            if _is_opt_role_op(op):
                # delete clip op from opt_ops when run in Parameter Server mode
                if OP_NAME_SCOPE in op.all_attrs() \
                        and CLIP_OP_NAME_SCOPE in op.attr(OP_NAME_SCOPE):
                    op._set_attr("op_role", role_id)
                    continue
                if op.attr(OP_ROLE_VAR_ATTR_NAME):
                    param_name = op.attr(OP_ROLE_VAR_ATTR_NAME)[0]
                    grad_name = op.attr(OP_ROLE_VAR_ATTR_NAME)[1]
                    if param_name not in optimize_params:
                        optimize_params.add(param_name)
                        param_grad = (origin_var_dict[param_name],
                                      origin_var_dict[grad_name])

                        if param_name in sparse_varnames:
                            sparse_param_grads.append(param_grad)
                        else:
                            dense_param_grads.append(param_grad)
        return sparse_param_grads, dense_param_grads

    def _get_sparse_varnames():
        sparse_varnames = []
        op_types = {"lookup_table": "W"}
        for op in origin_program.global_block().ops:
            if op.type in op_types.keys() \
                    and op.attr('remote_prefetch') is True:
                param_name = op.input(op.input_names[op_types[op.type]])[0]
                sparse_varnames.append(param_name)

        return list(set(sparse_varnames))

    def _append_send_op(union_vars, queue):
        send_input_vars = [
            program.global_block().vars[union_var] for union_var in union_vars
        ]

        dummy_output = []
        if mode in [DistributedMode.SYNC, DistributedMode.HALF_ASYNC]:
            dummy_output = program.global_block().create_var(
                name=framework.generate_control_dev_var_name())

        program.global_block().append_op(
            type="send",
            inputs={"X": send_input_vars},
            outputs={"Out": dummy_output},
            attrs={
                "queue": queue,
                "merge_add": True,
                "use_send_handler": False,
                RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE
            })

        return dummy_output

    def _append_barrier_op(dummys):
        program.global_block().append_op(
            type="send_barrier",
            inputs={"X": dummys},
            outputs={"Out": []},
            attrs={
                "endpoints": pserver_endpoints,
                "trainer_id": trainer_id,
                "half_async": True,
                RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE
            })

    sparse_varnames = _get_sparse_varnames()
    sparse_param_grads, dense_param_grads = _get_params_grads(sparse_varnames)

    dummys = []
    for sparse_union in sparse_param_grads:
        dummys.append(_append_send_op(sparse_union, "Q1"))
    for dense_union in dense_param_grads:
        dummys.append(_append_send_op(dense_union, "Q2"))

    if mode in [DistributedMode.SYNC, DistributedMode.HALF_ASYNC]:
        _append_barrier_op(dummys)

    return program


def lr_decay_pass(program):
    pass


def fake_init_ops_pass(program):
    return program


def get_communicator_context(program):
    send_context = []
    recv_context = []
    return send_context, recv_context
