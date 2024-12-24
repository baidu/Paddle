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

import unittest

import paddle
import paddle.nn.functional as F
from paddle import nn, static, utils
from paddle.distributed import fleet
from paddle.distributed.auto_parallel.static.completion import Completer
from paddle.distributed.auto_parallel.static.dist_context import (
    DistributedContext,
)
from paddle.distributed.auto_parallel.static.parallelizer import (
    AutoParallelizer,
)
from paddle.distributed.auto_parallel.static.partitioner import Partitioner
from paddle.distributed.auto_parallel.static.process_group import (
    ProcessGroup,
    _g_process_group_map,
)
from paddle.distributed.auto_parallel.static.reshard import Resharder
from paddle.distributed.fleet import auto

paddle.enable_static()
_global_parallel_strategy = None
_global_process_mesh = None
PP_MESH_0 = None
PP_MESH_1 = None


class MLPLayer(nn.Layer):
    def __init__(
        self,
        hidden_size=1024,
        intermediate_size=4 * 1024,
        initializer_range=0.02,
    ):
        super().__init__()
        d_model = hidden_size
        dim_feedforward = intermediate_size
        weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.Normal(mean=0.0, std=initializer_range)
        )
        bias_attr = None

        self.linear0 = nn.Linear(
            d_model, dim_feedforward, weight_attr, bias_attr=bias_attr
        )
        self.linear1 = nn.Linear(
            dim_feedforward, d_model, weight_attr, bias_attr=bias_attr
        )
        self.norm = nn.LayerNorm(d_model, epsilon=1e-5)

    def forward(self, input):
        if _global_parallel_strategy == "pp":
            auto.shard_tensor(self.linear0.weight, PP_MESH_0, [None, None])
            auto.shard_tensor(self.linear1.weight, PP_MESH_1, [None, None])
        else:
            auto.shard_tensor(
                self.linear0.weight, _global_process_mesh, [None, None]
            )
            auto.shard_tensor(
                self.linear1.weight, _global_process_mesh, [None, None]
            )

        out = self.norm(input)
        out = self.linear0(out)
        out = F.gelu(out, approximate=True)
        out = self.linear1(out)

        return out


def mlp_forward(train_program, start_program):
    with static.program_guard(
        train_program, start_program
    ), utils.unique_name.guard():
        batch_size = 4
        hidden_size = 1024
        sequence_len = 512
        input = static.data(
            name="input", shape=[batch_size, hidden_size], dtype='float32'
        )
        label = static.data(
            name="label", shape=[batch_size, 1], dtype='float32'
        )

        if _global_parallel_strategy == "pp":
            auto.shard_tensor(input, PP_MESH_0, [None, None])
            auto.shard_tensor(label, PP_MESH_1, [None, None])
        elif _global_parallel_strategy == "dp":
            auto.shard_tensor(input, _global_process_mesh, ["x", None])
        else:
            auto.shard_tensor(input, _global_process_mesh, [None, None])

        mlp = MLPLayer(
            hidden_size=hidden_size,
            intermediate_size=4 * hidden_size,
            initializer_range=0.02,
        )

        predict = mlp(input)
        error_cost = paddle.nn.functional.square_error_cost(predict, label)
        loss = paddle.mean(error_cost)

    return loss, train_program, start_program


def get_dist_prog(
    train_program,
    startup_program,
    dist_context,
    rank_id,
    change_process_mesh=False,
):
    loss, train_program, startup_program = mlp_forward(
        train_program, startup_program
    )

    fleet._user_defined_strategy = fleet.DistributedStrategy()
    fleet.user_defined_optimizer = paddle.optimizer.Adam()
    parallelizer = AutoParallelizer(fleet)
    parallelizer._dist_context = dist_context

    # serial forward & backward completion
    completer = Completer(dist_context)
    complete_train_program = completer.complete_forward_annotation(
        train_program
    )
    dist_context.block_state.parse_forward_blocks(complete_train_program)
    if change_process_mesh:
        global PP_MESH_1
        dist_context.get_tensor_dist_attr_for_program(
            train_program.global_block().vars["gelu_0.tmp_0"]
        ).process_mesh = PP_MESH_1

    params_grads = parallelizer._generate_backward(
        complete_train_program,
        startup_program,
        loss,
        parameter_list=None,
        no_grad_set=None,
        callbacks=None,
    )

    # logical partition
    partitioner = Partitioner(dist_context, rank_id)
    (
        auto_parallel_main_prog,
        auto_parallel_startup_prog,
        dist_params_grads,
    ) = partitioner.partition(
        complete_train_program, startup_program, params_grads
    )

    partitioned_optimize_ops = parallelizer._apply_optimize(
        auto_parallel_main_prog, auto_parallel_startup_prog, dist_params_grads
    )

    return (
        auto_parallel_main_prog,
        auto_parallel_startup_prog,
        dist_params_grads,
    )


def check_backward_dist_attr(dist_context, dist_main_prog, op_need_check):
    has_dist_attr = True
    vars = dist_main_prog.global_block().vars

    op_dist_attr = dist_context.get_op_dist_attr_for_program(op_need_check)
    if not op_dist_attr or not op_dist_attr.process_mesh:
        has_dist_attr = False

    for var_name in op_need_check.input_arg_names:
        if (
            not op_dist_attr.get_input_dims_mapping(var_name)
            or not dist_context.get_tensor_dist_attr_for_program(
                vars[var_name]
            ).dims_mapping
            or not dist_context.get_tensor_dist_attr_for_program(
                vars[var_name]
            ).process_mesh
        ):
            has_dist_attr = False
            break

    if has_dist_attr:
        for var_name in op_need_check.output_arg_names:
            if (
                not dist_context.get_tensor_dist_attr_for_program(
                    vars[var_name]
                ).dims_mapping
                or not dist_context.get_tensor_dist_attr_for_program(
                    vars[var_name]
                ).process_mesh
            ):
                has_dist_attr = False
                break

    return has_dist_attr


def check_send_recv_result(dist_main_prog, rank_id):
    send_result = False
    recv_result = False
    ops = dist_main_prog.global_block().ops

    if rank_id == 0:
        for idx, op in enumerate(ops):
            if op.type == "p_send" and "gelu_0.tmp_0" in op.input_arg_names:
                send_result = True
            if (
                op.type == "p_recv"
                and "gelu_0.tmp_0@GRAD" in op.output_arg_names[0]
            ):
                recv_result = True
    else:
        for idx, op in enumerate(ops):
            if (
                op.type == "p_send"
                and "gelu_0.tmp_0@GRAD" in op.input_arg_names
            ):
                send_result = True
            if op.type == "p_recv" and "gelu_0.tmp_0" in op.output_arg_names[0]:
                recv_result = True

    return send_result and recv_result


def check_initialization(dist_startup_prog, rank_id):
    if rank_id == 0:
        need_check_params = [
            "layer_norm_0.b_0",
            "layer_norm_0.w_0",
            "linear_0.w_0",
            "linear_0.b_0",
        ]
    else:
        need_check_params = ['linear_1.w_0', 'linear_1.b_0']

    params = []
    for var_name, var in dist_startup_prog.global_block().vars.items():
        if var.is_parameter:
            params.append(var_name)

    return params == need_check_params


def check_initialization_for_dp(dist_startup_prog):
    need_check_params = [
        "layer_norm_0.b_0",
        "layer_norm_0.w_0",
        "linear_0.w_0",
        "linear_0.b_0",
        "linear_1.w_0",
        "linear_1.b_0",
    ]
    params = []
    for var_name, var in dist_startup_prog.global_block().vars.items():
        if var.is_parameter:
            params.append(var_name)
    broadcast_varnames = []
    for op in dist_startup_prog.global_block().ops:
        if op.type == "broadcast":
            broadcast_varnames.append(op.output_arg_names[0])

    return (
        sorted(params)
        == sorted(need_check_params)
        == sorted(broadcast_varnames)
    )


class TestMLPReshard(unittest.TestCase):
    def test_complete_backward_annotation(self):
        global _global_process_mesh
        _global_process_mesh = auto.ProcessMesh(mesh=[0, 1])

        train_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        dist_context = DistributedContext()
        rank_id = 0
        dist_main_prog, dist_startup_prog, dist_params_grads = get_dist_prog(
            train_program, startup_program, dist_context, 0
        )

        op_need_check = None
        for op in dist_main_prog.global_block().ops:
            if op.type == "gelu_grad":
                op_need_check = op
                break

        # grad op should have dist attr
        self.assertTrue(
            check_backward_dist_attr(
                dist_context, dist_main_prog, op_need_check
            )
        )

        # clear _g_process_group_map
        _g_process_group_map.clear()
        _g_process_group_map[0] = ProcessGroup(0, [])

    def test_mlp_pp(self):
        global _global_parallel_strategy
        _global_parallel_strategy = "pp"
        global _global_process_mesh
        _global_process_mesh = auto.ProcessMesh(mesh=[0, 1], dim_names=["x"])
        global PP_MESH_0
        PP_MESH_0 = auto.ProcessMesh(mesh=[0], dim_names=["x"])
        global PP_MESH_1
        PP_MESH_1 = auto.ProcessMesh(mesh=[1], dim_names=["x"])

        train_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        dist_context = DistributedContext()
        rank_id = 1
        dist_main_prog, dist_startup_prog, dist_params_grads = get_dist_prog(
            train_program, startup_program, dist_context, rank_id
        )
        resharder = Resharder(
            dist_main_prog,
            dist_startup_prog,
            rank_id,
            dist_context,
            dist_params_grads,
        )
        resharder.reshard()

        # check send and recv result
        self.assertTrue(check_send_recv_result(dist_main_prog, rank_id))
        # parameter initialization of every rank should be different in the pipeline scene
        self.assertTrue(check_initialization(dist_startup_prog, rank_id))

        # clear _g_process_group_map
        _g_process_group_map.clear()
        _g_process_group_map[0] = ProcessGroup(0, [])

    def test_mlp_pp_diff_process_mesh(self):
        global _global_parallel_strategy
        _global_parallel_strategy = "pp"
        global _global_process_mesh
        _global_process_mesh = auto.ProcessMesh(mesh=[0, 1], dim_names=["x"])
        global PP_MESH_0
        PP_MESH_0 = auto.ProcessMesh(mesh=[0], dim_names=["x"])
        global PP_MESH_1
        PP_MESH_1 = auto.ProcessMesh(mesh=[1], dim_names=["x"])

        train_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        dist_context = DistributedContext()
        rank_id = 1
        dist_main_prog, dist_startup_prog, dist_params_grads = get_dist_prog(
            train_program, startup_program, dist_context, rank_id, True
        )
        resharder = Resharder(
            dist_main_prog,
            dist_startup_prog,
            rank_id,
            dist_context,
            dist_params_grads,
        )
        resharder.reshard()
        # check send and recv result
        self.assertTrue(check_send_recv_result(dist_main_prog, rank_id))
        self.assertTrue(check_initialization(dist_startup_prog, rank_id))

        # clear _g_process_group_map
        _g_process_group_map.clear()
        _g_process_group_map[0] = ProcessGroup(0, [])

    def test_mlp_dp(self):
        global _global_parallel_strategy
        _global_parallel_strategy = "dp"
        global _global_process_mesh
        _global_process_mesh = auto.ProcessMesh(mesh=[0, 1], dim_names=["x"])

        train_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        dist_context = DistributedContext()
        rank_id = 0
        dist_main_prog, dist_startup_prog, dist_params_grads = get_dist_prog(
            train_program, startup_program, dist_context, rank_id
        )
        resharder = Resharder(
            dist_main_prog,
            dist_startup_prog,
            rank_id,
            dist_context,
            dist_params_grads,
        )
        resharder.reshard()

        # send and recv should not exist in dp scene.
        self.assertFalse(check_send_recv_result(dist_main_prog, rank_id))
        # all parameters should be initialized in dp scene
        self.assertTrue(check_initialization_for_dp(dist_startup_prog))

        # clear _g_process_group_map
        _g_process_group_map.clear()
        _g_process_group_map[0] = ProcessGroup(0, [])


if __name__ == "__main__":
    unittest.main()
