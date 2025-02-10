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

import collections
import logging

from ..auto_parallel.static.utils import (
    get_logger,
)
from .pass_base import PassBase, register_pass
from .pass_utils import AutoParallelStreamType, split_matmul_grad_to_matmul

logger = get_logger(logging.INFO)


# For allreduce pattern in the backward phase of column parallel linear:
#   dX, dY = matmul_grad(X, Y, dOut)
#   dX = c_allreduce_sum(dX)
# Split matmul_grad to 2 matmul:
#   dX = matmul(dOut, Y^T)
#   dX = c_allreduce_sum(dX)
#   dY = matmul(X^T, dOut)
#
# Then the all_reduce sum can overlap with the compute of dY.
@register_pass("allreduce_matmul_grad_overlapping")
class AllreduceMatmulGradOverlappingPass(PassBase):
    def __init__(self):
        super().__init__()
        self.op_namescope = "/auto_parallel/allreduce_matmul_grad_overlapping"
        self.set_attr("dist_context", None)

    def _check_self(self):
        if self.get_attr("dist_context") is None:
            return False
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, context):
        self.dist_context = self.get_attr("dist_context")
        block = main_program.global_block()

        matmul_grad_id_to_allreduce_id = (
            self._get_all_matmul_grad_and_allreduce_pairs(block)
        )
        logger.info(
            f"overlap matmul_grad and allreduce: {matmul_grad_id_to_allreduce_id}"
        )

        self._split_matmul_grad_and_multi_streaming_allreduce(
            block, matmul_grad_id_to_allreduce_id
        )

    def _get_all_matmul_grad_and_allreduce_pairs(self, block):
        ops = block.ops
        op_num = len(ops)
        matmul_grad_id_to_allreduce_id = collections.OrderedDict()
        for i, op_i in enumerate(ops):
            if (
                op_i.type == 'matmul_v2_grad'
                and op_i.attr("trans_x") is False
                and op_i.attr("trans_y") is False
            ):
                x_grad = op_i.output("X@GRAD")
                for j in range(i + 1, op_num):
                    op_j = ops[j]
                    if (
                        op_j.type == 'c_allreduce_sum'
                        and op_j.input("X") == x_grad
                    ):
                        matmul_grad_id_to_allreduce_id[i] = j
        return matmul_grad_id_to_allreduce_id

    def _split_matmul_grad_and_multi_streaming_allreduce(
        self, block, matmul_grad_id_to_allreduce_id
    ):
        ops = block.ops

        for matmul_grad_id, allreduce_id in reversed(
            matmul_grad_id_to_allreduce_id.items()
        ):
            matmul_grad_op = ops[matmul_grad_id]
            allreduce_op = ops[allreduce_id]

            # NOTE(Sonder): When there are ops between matmul_grad and allreduce, we should check whether
            # these ops rely on the output of the intermediate ops. If so, we should not split the matmul_grad.
            # Otherwise, the output of the intermediate ops will get wrong results.
            skip_overlapping = False
            moved_ops_output = []
            matmul_grad_output = matmul_grad_op.output('Y@GRAD')[0]

            for idx in range(matmul_grad_id + 1, allreduce_id):
                if matmul_grad_output in ops[idx].desc.input_arg_names():
                    moved_ops_output.extend(ops[idx].desc.output_arg_names())
                else:
                    for input_name in ops[idx].desc.input_arg_names():
                        if input_name in moved_ops_output:
                            skip_overlapping = True

            if skip_overlapping:
                continue

            # matmul_grad_op => matmul_v2 + reshape + reshape + matmul_v2 + reshape
            split_matmul_grad_to_matmul(
                block, matmul_grad_id, self.dist_context, self.op_namescope
            )

            # NOTE(Ruibiao): Required OP scheduling order: matmul(dOut, Y^T) -> c_allreduce_sum(dX) -> matmul(X^T, dOut).
            # c_allreduce_sum(dX) and matmul(X^T, dOut) cannot be swapped. Otherwise, after buffer_shared_inplace_pass
            # adding share_buffer OP before c_allreduce_sum, c_allreduce_sum will synchronous with comp-stream, and then
            # the matmul op before it cannot be overlapped.
            allreduce_op_dist_attr = (
                self.dist_context.get_op_dist_attr_for_program(allreduce_op)
            )
            allreduce_op_dist_attr.execution_stream = (
                AutoParallelStreamType.MP_STREAM.value
            )

            allreduce_op_inputs = allreduce_op.desc.input_names()
            allreduce_op_outputs = allreduce_op.desc.output_names()

            allreduce_op_inputs = {
                name: allreduce_op.input(name) for name in allreduce_op_inputs
            }
            allreduce_op_outputs = {
                name: allreduce_op.output(name) for name in allreduce_op_outputs
            }

            # matmul_v2 + reshape + reshape + matmul_v2 + reshape + ... + original c_allreduce_sum
            # =>
            # matmul_v2 + new c_allreduce_sum + reshape + reshape + matmul_v2 + reshape + ... + original c_allreduce_sum
            #
            # NOTE(liym27): new c_allreduce_sum must be inserted to "the next of the first matmul_v2", otherwise another
            # pass fused_linear_param_grad_add will not work.
            allreduce_op = block._insert_op_without_sync(
                index=matmul_grad_id + 1,
                type=allreduce_op.type,
                inputs=allreduce_op_inputs,
                outputs=allreduce_op_outputs,
                attrs=allreduce_op.all_attrs(),
            )
            self.dist_context.set_op_dist_attr_for_program(
                allreduce_op, allreduce_op_dist_attr
            )
            # Remove the original allreduce op
            block._remove_op(allreduce_id + 5, sync=False)

        block._sync_with_cpp()
