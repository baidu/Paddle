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
from paddle.distributed.auto_parallel.static.cluster import Cluster
from paddle.distributed.auto_parallel.static.completion import Completer
from paddle.distributed.auto_parallel.static.cost import CostEstimator
from paddle.distributed.auto_parallel.static.dist_context import (
    DistributedContext,
)
from paddle.distributed.auto_parallel.static.parallelizer import (
    AutoParallelizer,
)
from paddle.distributed.auto_parallel.static.partitioner import Partitioner
from paddle.distributed.auto_parallel.static.reshard import Resharder
from paddle.distributed.fleet import auto

paddle.enable_static()
_global_parallel_strategy = "mp_pp"
_global_process_mesh = auto.ProcessMesh([[0, 1], [2, 3]], dim_names=["x", "y"])
PP_MESH_0 = auto.ProcessMesh([0, 1], dim_names=["x"])
PP_MESH_1 = auto.ProcessMesh([2, 3], dim_names=["x"])


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

        self.word_embeddings = nn.Embedding(
            hidden_size,
            hidden_size,
            weight_attr=paddle.ParamAttr(
                name="word_embeddings",
                initializer=nn.initializer.Normal(
                    mean=0.0, std=initializer_range
                ),
            ),
        )

        self.linear0 = nn.Linear(
            d_model, dim_feedforward, weight_attr, bias_attr=bias_attr
        )
        self.linear1 = nn.Linear(
            dim_feedforward, d_model, weight_attr, bias_attr=bias_attr
        )
        self.linear2 = nn.Linear(
            dim_feedforward, d_model, weight_attr, bias_attr=bias_attr
        )

    def forward(self, input):
        auto.shard_tensor(self.word_embeddings.weight, PP_MESH_0, ["x", None])
        auto.shard_tensor(self.linear0.weight, PP_MESH_0, [None, "x"])
        auto.shard_tensor(self.linear1.weight, PP_MESH_1, ["x", None])
        auto.shard_tensor(self.linear2.weight, PP_MESH_1, ["x", None])
        w_out = self.word_embeddings(input)
        out = self.linear0(w_out)
        param = paddle.create_parameter([4096, 4096], paddle.float32)
        auto.shard_tensor(param, PP_MESH_0, ["x", None])
        out = paddle.matmul(out, param)
        gelu_out = F.gelu(out, approximate=True)
        out = self.linear1(gelu_out)
        out1 = self.linear2(gelu_out)
        out = out + out1

        return out


def mlp_forward(train_program, start_program):
    with static.program_guard(
        train_program, start_program
    ), utils.unique_name.guard():
        batch_size = 4
        hidden_size = 1024
        sequence_len = 512
        input = static.data(name="input", shape=[batch_size], dtype='int32')
        label = static.data(
            name="label", shape=[batch_size, 1], dtype='float32'
        )

        auto.shard_tensor(input, PP_MESH_0, [None])
        auto.shard_tensor(label, PP_MESH_1, [None, None])

        mlp = MLPLayer(
            hidden_size=hidden_size,
            intermediate_size=4 * hidden_size,
            initializer_range=0.02,
        )

        predict = mlp(input)
        error_cost = paddle.nn.functional.square_error_cost(predict, label)
        loss = paddle.mean(error_cost)

    return loss, train_program, start_program


def get_dist_prog(train_program, startup_program, dist_context, rank_id):
    global _global_process_mesh
    dist_context.process_mesh = _global_process_mesh
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


def check_send_recv_result(dist_main_prog, rank_id):
    send_result = False
    recv_result = False
    ops = dist_main_prog.global_block().ops
    if rank_id in [0, 1]:
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
                and "gelu_0.tmp_0@GRAD" in op.input_arg_names[0]
            ):
                send_result = True
            if op.type == "p_recv" and "gelu_0.tmp_0" in op.output_arg_names[0]:
                recv_result = True

    return send_result and recv_result


def check_initialization_for_mppp(dist_startup_prog, rank_id):
    if rank_id in [0, 1]:
        need_check_params = []
    else:
        need_check_params = ["linear_1.b_0", "linear_2.b_0"]
    broadcast_varnames = []
    for op in dist_startup_prog.global_block().ops:
        if op.type == "broadcast":
            broadcast_varnames.append(op.output_arg_names[0])

    return need_check_params == broadcast_varnames


def check_allgather(dist_main_program):
    allgather_out = "all_gather@RESHARD_0.tmp_0"  # "x@RESHARD_0"
    var_result = False
    op_result = False
    vars = dist_main_program.global_block().vars
    if allgather_out in vars and vars[allgather_out].shape == (4, 4):
        var_result = True
    for op in dist_main_program.global_block().ops:
        if op.type == "matmul_v2":
            if allgather_out in op.input_arg_names:
                op_result = True
    return var_result and op_result


class TestMLPReshard(unittest.TestCase):
    def test_mlp_mppp(self):
        train_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        dist_context = DistributedContext()
        rank_id = 2
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

        # parameter which not been sliced should be the same in the mp scene
        self.assertTrue(
            check_initialization_for_mppp(dist_startup_prog, rank_id)
        )

    def test_allgather(self):
        train_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        process_mesh = auto.ProcessMesh(mesh=[0, 1], dim_names=["x"])
        with static.program_guard(train_program, startup_program):
            x = paddle.static.data(name="x", shape=[4, 4], dtype='float32')
            x = auto.shard_tensor(x, process_mesh, ["x", None])
            w = paddle.static.data(name="w", shape=[4, 4], dtype='float32')
            w = auto.shard_tensor(w, process_mesh, [None, None])

            y = paddle.distributed.shard_op(
                paddle.matmul, process_mesh, [[None, None], [None, None]]
            )(x, w)

        rank_id = 0
        dist_context = DistributedContext()
        dist_strategy = fleet.DistributedStrategy()
        partitioner = Partitioner(dist_context, rank_id)
        completer = Completer(dist_context)
        complete_train_program = completer.complete_forward_annotation(
            train_program
        )
        dist_context.block_state.parse_forward_blocks(complete_train_program)
        (
            partitioned_main_prog,
            partitioned_startup_prog,
            partitioned_params_grads,
        ) = partitioner.partition(complete_train_program, startup_program, [])

        # test estimator
        cluster = Cluster()
        cluster.gen_default_config_cluster(device_count=2)
        cost_estimator = CostEstimator(train_program, cluster)
        global_cost = cost_estimator.estimate(dist_context)
        max_memory = cost_estimator._estimate_max_memory_by_dist_op(
            dist_context
        )
        # test cache
        global_cost = cost_estimator.estimate(dist_context)
        max_memory = cost_estimator._estimate_max_memory_by_dist_op(
            dist_context
        )
        assert global_cost.time > 0
        assert max_memory > 0

        resharder = Resharder(
            partitioned_main_prog,
            partitioned_startup_prog,
            rank_id,
            dist_context,
            partitioned_params_grads,
        )
        resharder.reshard()
        # the x should not be slice
        self.assertTrue(check_allgather(partitioned_main_prog))

    def test_c_concat(self):
        train_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        process_mesh = auto.ProcessMesh(mesh=[0, 1], dim_names=["x"])
        with static.program_guard(train_program, startup_program):
            x = paddle.static.data(name="x", shape=[4, 4], dtype='float32')
            x = auto.shard_tensor(x, process_mesh, [None, "x"])
            w = paddle.static.data(name="w", shape=[4, 4], dtype='float32')
            w = auto.shard_tensor(w, process_mesh, [None, None])

            y = paddle.distributed.shard_op(
                paddle.matmul, process_mesh, [[None, None], [None, None]]
            )(x, w)

        rank_id = 0
        dist_context = DistributedContext()
        dist_strategy = fleet.DistributedStrategy()
        partitioner = Partitioner(dist_context, rank_id)
        completer = Completer(dist_context)
        complete_train_program = completer.complete_forward_annotation(
            train_program
        )
        dist_context.block_state.parse_forward_blocks(complete_train_program)
        (
            partitioned_main_prog,
            partitioned_startup_prog,
            partitioned_params_grads,
        ) = partitioner.partition(complete_train_program, startup_program, [])

        # test estimator
        cluster = Cluster()
        cluster.gen_default_config_cluster(device_count=2)
        cost_estimator = CostEstimator(train_program, cluster)
        global_cost = cost_estimator.estimate(dist_context)
        max_memory = cost_estimator._estimate_max_memory_by_dist_op(
            dist_context
        )
        # test cache
        global_cost = cost_estimator.estimate(dist_context)
        max_memory = cost_estimator._estimate_max_memory_by_dist_op(
            dist_context
        )
        assert global_cost.time >= 0
        assert max_memory > 0

        resharder = Resharder(
            partitioned_main_prog,
            partitioned_startup_prog,
            rank_id,
            dist_context,
            partitioned_params_grads,
        )
        resharder.reshard()


if __name__ == "__main__":
    unittest.main()
