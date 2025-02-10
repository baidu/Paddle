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

import paddle

from .reshard_funcs.base_reshard_func import is_replicated
from .utils import _complete_op_dist_attr

dist_skip_op_list = [
    "builtin.combine",
    "builtin.split",
    "cf.yield",
    "cf.tuple_push",
    "cf.tuple_pop",
    "cf.stack_create",
    "pd_op.pylayer",
]


def verify_dist_block(block):
    for op in block.ops:
        if op.name() in dist_skip_op_list:
            continue
        if op.name() == "dist_op.shard_tensor":
            raise RuntimeError("Block still contain shard_tensor_op.")
        # Note (luchang): Temp fix, remove unused parameter 'op'.
        # Will be removed in the future.
        if op.name() == "builtin.parameter":
            if op.result(0).use_empty():
                op.erase()
                continue


def apply_mix2dist_pass(program, block=None):
    if block is None:
        block = program.global_block()
    deleted_ops = []
    for op in block.ops:
        for inner_block in op.blocks():
            apply_mix2dist_pass(program, block=inner_block)
        if op.name() != "dist_op.shard_tensor":
            continue
        shard_operand_value = op.operand_source(0)
        if not shard_operand_value.has_one_use():
            raise RuntimeError(
                f"shard_tensor is supposed to be called right after tensor is created, the use_count of tensor to be sharded is {shard_operand_value.use_count}, which is "
                "not Supported in right now."
            )
        shard_result_value = op.result(0)
        shard_result_value.replace_all_uses_with(shard_operand_value)
        deleted_ops.append(op)
        prev_op = shard_operand_value.get_defining_op()
        if (
            prev_op.name() == "builtin.parameter"
            or prev_op.name() == "pd_op.data"
        ):
            prev_op.dist_attr = op.dist_attr
            shard_operand_value.set_type(shard_result_value.type())
            shard_operand_value.stop_gradient = shard_result_value.stop_gradient
            shard_operand_value.persistable = shard_result_value.persistable
        elif (
            prev_op.name() == "pd_op.randint"
            or prev_op.name() == "pd_op.gaussian"
        ):
            mesh = shard_result_value.dist_attr().process_mesh
            # input
            shape_value = prev_op.operand_source(0)
            dist_attr = paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                mesh, [-1 for _ in range(len(shape_value.shape))], {}
            )
            shape_value.update_dist_attr(dist_attr)
            # op
            prev_op.dist_attr = (
                paddle.base.libpaddle.pir.create_op_dist_attribute(
                    mesh, [dist_attr], [shard_result_value.dist_attr()]
                )
            )
            # deal with full_int_array op
            prev_prev_op = shape_value.get_defining_op()
            prev_prev_op.dist_attr = (
                paddle.base.libpaddle.pir.create_op_dist_attribute(
                    mesh, [], [dist_attr]
                )
            )
            # output
            shard_operand_value.set_type(shard_result_value.type())
            shard_operand_value.stop_gradient = shard_result_value.stop_gradient
            shard_operand_value.persistable = shard_result_value.persistable
        else:
            dist_attr = shard_result_value.dist_attr()
            if not is_replicated(dist_attr):
                raise RuntimeError(
                    f"{prev_op} is not support sharded by shard_tensor op in pir mode."
                )
            mesh = dist_attr.process_mesh
            ops_list = [prev_op]
            while len(ops_list) != 0:
                cur_op = ops_list.pop()
                if cur_op.dist_attr is not None:
                    continue
                operand_attrs = []
                result_attrs = []
                for input in cur_op.operands_source():
                    dist_attr = (
                        paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                            mesh, [-1 for _ in range(len(input.shape))], {}
                        )
                    )
                    operand_attrs.append(dist_attr)
                    ops_list.append(input.get_defining_op())
                for result in cur_op.results():
                    dist_attr = (
                        paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                            mesh, [-1 for _ in range(len(result.shape))], {}
                        )
                    )
                    result.update_dist_attr(dist_attr)
                    result_attrs.append(dist_attr)
                cur_op.dist_attr = (
                    paddle.base.libpaddle.pir.create_op_dist_attribute(
                        mesh, operand_attrs, result_attrs
                    )
                )
    for op in deleted_ops:
        op.erase()
    _complete_op_dist_attr(program, block=block)
    verify_dist_block(block)
