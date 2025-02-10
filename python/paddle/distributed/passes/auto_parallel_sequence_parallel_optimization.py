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


import paddle
from paddle.distributed.auto_parallel.static.utils import (
    naive_set_dist_op_attr_for_program_by_mesh,
)
from paddle.distributed.fleet.meta_optimizers.common import OP_ROLE_KEY
from paddle.distributed.utils.stream_utils import ExecutionStreamType
from paddle.static import default_main_program

from .auto_parallel_sharding import _is_reshard_op
from .pass_base import PassBase, PassType, register_pass


# NOTE we add the "auto_parallel" prefix to the pass in order to
# indicate that this pass should obey some constrains by auto_parallel
# for example all ops and vars should has dist attr before and after pass
# should use dist op instead of custom comm op
@register_pass("auto_parallel_sequence_parallel_optimization")
class SequenceParallelOptimizationPass(PassBase):
    """
    This pass is used to optimize the sequence parallel.
        1. Fuse the allreduce + split into reducescatter.
        2. Trade off communication for memory in the row_parallel_linear output.
        3. Overlap communication with computation in backward computation.
    """

    def __init__(self):
        super().__init__()
        self.set_attr("dist_context", None)

    def _check_self(self):
        if self.get_attr("dist_context") is None:
            return False
        if (not isinstance(self.get_attr("global_rank"), int)) or self.get_attr(
            "global_rank"
        ) < 0:
            return False
        if not self.get_attr("dist_context").strategy.sp_optimization.enable:
            return False
        return True

    def _check_conflict(self, other_pass):
        return True

    def _type(self):
        return PassType.COMM_OPT

    def _apply_single_impl(self, main_program, startup_program, context):
        self.dist_context = self.get_attr("dist_context")
        self.global_rank = int(self.get_attr("global_rank"))

        with paddle.static.program_guard(main_program, startup_program):
            # TODO remove this pass when we use local reshard for all communication
            self._fuse_allreduce_split()
            self._memory_optimization()
            self._overlap()

    def _fuse_allreduce_split(self):
        # allreduce is added by dist op and split is added by reshard, so we need this pass to fuse them as reducescatter.
        # reducescatter should be inferred by local reshard in future.

        block = default_main_program().global_block()

        # record valid split ops
        valid_split_op_indices = []

        def is_valid_split_op(idx, block):
            op = block.ops[idx]
            if not op.type == "split":
                return False
            pre_op = block.ops[idx - 1]
            if not pre_op.type == "c_allreduce_sum":
                return False
            pre_output_name = pre_op.output_arg_names[0]
            cur_input_name = op.input_arg_names[0]
            if (
                pre_output_name == cur_input_name
                and _is_reshard_op(op)
                and op.attr("axis") == 0
            ):
                return True
            return False

        for i in range(len(block.ops)):
            if is_valid_split_op(i, block):
                valid_split_op_indices.append(i)

        # modify program
        remove_varnames = []
        for i in sorted(valid_split_op_indices, reverse=True):
            allreduce_op = block.ops[i - 1]
            split_op = block.ops[i]
            consumer_op = block.ops[i + 1]

            allreduce_input_name = allreduce_op.input("X")[0]
            ring_id = int(allreduce_op.attr("ring_id"))
            split_output_names = split_op.output("Out")
            nranks = len(split_output_names)
            consumer_input_names = consumer_op.input_arg_names
            intersection = set(split_output_names).intersection(
                set(consumer_input_names)
            )
            assert (
                len(intersection) == 1
            ), f"Sequence Parallel ReduceScatter Output more than 1: {intersection}."
            keep_output_name = intersection.pop()
            split_output_names.remove(keep_output_name)
            remove_varnames.extend(split_output_names)

            # replace ops
            new_op = block._insert_op_without_sync(
                index=i + 1,
                type="reduce_scatter",
                inputs={'x': [allreduce_input_name]},
                outputs={'out': [keep_output_name]},
                attrs={
                    'ring_id': ring_id,
                    'nranks': nranks,
                    'op_namescope': allreduce_op.attr("op_namescope"),
                    OP_ROLE_KEY: consumer_op.attr(OP_ROLE_KEY),
                },
            )
            new_op.dist_attr.execution_stream = (
                ExecutionStreamType.DefaultStream.value
            )
            block._remove_op(i, False)
            block._remove_op(i - 1, False)

            # set dist attr
            allreduce_input_dist_attr = (
                self.dist_context.get_tensor_dist_attr_for_program(
                    block.vars[allreduce_input_name]
                )
            )
            ref_process_mesh = allreduce_input_dist_attr.process_mesh
            naive_set_dist_op_attr_for_program_by_mesh(
                new_op,
                ref_process_mesh,
                self.dist_context,
                chunk_id=allreduce_input_dist_attr.chunk_id,
            )

        # remove vars
        for varname in remove_varnames:
            block._remove_var(varname, sync=False)

        block._sync_with_cpp()

    def _memory_optimization(self):
        pass

    def _overlap(self):
        pass
