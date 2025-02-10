# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.distributed as dist
from paddle.base.framework import (
    auto_complete_op_role,
    pir_chunk_id_guard,
    pir_op_role_guard,
)
from paddle.distributed import Replicate, Shard
from paddle.distributed.auto_parallel.static.mix_to_dist_pass import (
    apply_mix2dist_pass,
)
from paddle.distributed.auto_parallel.static.pir_pass import (
    ReshardPasses,
    apply_partition_pass,
)


class TestOpRole(unittest.TestCase):
    def test_single(self):
        paddle.enable_static()
        with paddle.pir_utils.IrGuard():
            main_program = paddle.base.Program()
            with paddle.base.program_guard(main_program):

                # op_role = -1
                x0 = paddle.static.data(name='x0', shape=[1, 128, 512])
                x1 = paddle.nn.functional.relu(x0)
                x2 = paddle.nn.functional.relu(x1)

                with pir_op_role_guard(1), pir_chunk_id_guard(2):
                    y0 = paddle.static.data(name='y0', shape=[1, 128, 512])
                    y1 = paddle.nn.functional.relu(y0)
                    z0 = paddle.add(y1, x2)
                    z0 = z0 * 3.0
                with pir_op_role_guard(3), pir_chunk_id_guard(1):
                    z1 = paddle.nn.functional.relu(z0)
                    z2 = paddle.add(y0, z1)
                    z4 = paddle.split(z0, num_or_sections=[8, 100, 20], axis=1)

                with pir_op_role_guard(0), pir_chunk_id_guard(3):
                    z3 = paddle.add(y1, z2)

                # op_role = -1
                z4 = paddle.add(y0, z3)

        # check global shape
        std_ops = [
            "pd_op.data:-1:-1",
            "pd_op.data:1:2",
            "pd_op.relu:-1:-1",
            "pd_op.relu:-1:-1",
            "pd_op.relu:1:2",
            "pd_op.add:1:2",
            "pd_op.full:1:2",
            "pd_op.scale:1:2",
            "pd_op.relu:3:1",
            "pd_op.add:3:1",
            "pd_op.full_int_array:3:1",
            "pd_op.full:3:1",
            "pd_op.split:3:1",
            "builtin.split:3:1",
            "pd_op.add:0:3",
            "pd_op.add:-1:-1",
        ]

        cur_ops = [
            f"{op.name()}:{op.op_role}:{op.chunk_id}"
            for op in main_program.global_block().ops
        ]
        self.assertEqual(cur_ops, std_ops)

    def test_dist(self):
        paddle.enable_static()

        mesh = dist.ProcessMesh([[0, 1], [2, 3]], dim_names=["x", "y"])

        with paddle.pir_utils.IrGuard():
            main_program = paddle.base.Program()
            with paddle.base.program_guard(main_program):

                with auto_complete_op_role(main_program, 0):
                    x0 = paddle.static.data(name='x0', shape=[1, 128, 512])
                    x0 = dist.shard_tensor(
                        x0, mesh, [Shard(1), Replicate()], stop_gradient=False
                    )
                    x1 = x0 / 2.0

                    with pir_op_role_guard(3):
                        x2 = dist.reshard(x1, mesh, [Shard(2), Replicate()])
                        with pir_op_role_guard(1):
                            x3 = dist.reshard(
                                x2, mesh, [Replicate(), Replicate()]
                            )
                        x4 = dist.reshard(x3, mesh, [Shard(1), Replicate()])

                    x5 = dist.reshard(x4, mesh, [Replicate(), Replicate()])

        apply_mix2dist_pass(main_program)
        apply_partition_pass(main_program)
        ReshardPasses.apply_reshard_pass(main_program)

        std_ops = [
            'pd_op.data:0',
            'pd_op.full:0',
            'pd_op.scale:0',
            'pd_op.all_gather:3',
            'pd_op.full:3',
            'pd_op.split_with_num:3',
            'pd_op.full:3',
            'pd_op.concat:3',
            'pd_op.full_int_array:3',
            'pd_op.full_int_array:3',
            'pd_op.slice:3',
            'pd_op.all_gather:1',
            'pd_op.full:1',
            'pd_op.split_with_num:1',
            'pd_op.full:1',
            'pd_op.concat:1',
            'pd_op.full_int_array:3',
            'pd_op.full_int_array:3',
            'pd_op.slice:3',
            'pd_op.all_gather:0',
            'pd_op.full:0',
            'pd_op.split_with_num:0',
            'pd_op.full:0',
            'pd_op.concat:0',
        ]

        cur_ops = [
            f"{op.name()}:{op.op_role}"
            for op in main_program.global_block().ops
        ]

        self.assertEqual(cur_ops, std_ops)


if __name__ == "__main__":
    unittest.main()
