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

import os

import paddle
import paddle.distributed as dist
from paddle.distributed.auto_parallel.static.mix_to_dist_pass import (
    apply_mix2dist_pass,
)
from paddle.distributed.auto_parallel.static.pir_pass import (
    RemovePasses,
    ReshardPasses,
    apply_partition_pass,
)
from paddle.distributed.auto_parallel.static.utils import set_all_ops_op_role
from paddle.distributed.fleet.meta_optimizers.common import OpRole


class TestReshardRToSCrossMesh:
    def __init__(self):
        self._shape = eval(os.getenv("shape"))
        self._dtype = os.getenv("dtype")
        self._seeds = eval(os.getenv("seeds"))
        self._shard = eval(os.getenv("shard"))
        self._backend = os.getenv("backend")
        self._in_mesh = dist.ProcessMesh([0, 2], dim_names=["x"])
        self._out_mesh = dist.ProcessMesh([1, 3], dim_names=["x"])

    def run_test_case(self):
        paddle.enable_static()

        BATCH_SIZE = 2
        SEQ_LEN = 4
        HIDDEN_SIZE = 8

        with paddle.pir_utils.IrGuard():
            main_program = paddle.base.Program()
            with paddle.base.program_guard(main_program):
                input = paddle.static.data(
                    name='input', shape=[BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE]
                )
                input_tensor = dist.shard_tensor(
                    input, self._in_mesh, [dist.Replicate()]
                )
                out = paddle._C_ops.reshard(
                    input_tensor, self._out_mesh, [dist.Shard(self._shard)]
                )
            target_type = out.type()

            old_ops = [op.name() for op in main_program.global_block().ops]
            assert 'dist_op.reshard' in old_ops

            apply_mix2dist_pass(main_program)
            set_all_ops_op_role(main_program.global_block(), OpRole.Forward)
            apply_partition_pass(main_program)
            ReshardPasses.apply_reshard_pass(main_program)
            RemovePasses.remove_other_rank_op_pass(main_program)

        # np.testing.assert_equal(dist_program.num_ops(), 6)
        new_ops = [op.name() for op in main_program.global_block().ops]
        assert 'dist_op.reshard' not in new_ops
        if dist.get_rank() in self._in_mesh.process_ids:
            assert 'pd_op.p_send' in new_ops
        else:
            assert 'pd_op.p_recv' in new_ops
            assert 'pd_op.slice' in new_ops
        for op in main_program.global_block().ops:
            if op.name() == 'pd_op.p_send':
                assert op.dist_attr.process_mesh == self._in_mesh
                assert op.operand_source(0).dist_attr() == op.dist_attr.operand(
                    0
                )

                operand_dist_attr = op.operand_source(0).dist_attr()
                assert operand_dist_attr.process_mesh == self._in_mesh
                assert operand_dist_attr.dims_mapping == [-1, -1, -1]
                assert operand_dist_attr.partial_status == {}
            elif op.name() == 'pd_op.p_recv':
                assert op.dist_attr.process_mesh == self._out_mesh
                assert op.result(0).dist_attr() == op.dist_attr.result(0)
                result_dist_attr = op.result(0).dist_attr()
                assert result_dist_attr.process_mesh == self._out_mesh
                assert result_dist_attr.dims_mapping == [-1, -1, -1]
                assert result_dist_attr.partial_status == {}
            elif op.name() == 'pd_op.slice':
                assert op.dist_attr.process_mesh == self._out_mesh
                assert op.result(0).dist_attr() == op.dist_attr.result(0)
                assert op.result(0).type() == target_type


if __name__ == '__main__':
    TestReshardRToSCrossMesh().run_test_case()
