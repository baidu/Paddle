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
from __future__ import print_function
import unittest
import copy

import numpy as np

import paddle
import paddle.nn as nn
import paddle.static as static
import paddle.nn.functional as F
import paddle.utils as utils
import paddle.fluid.core as core
from paddle.distributed.auto_parallel.operators.common import get_distributed_operator_impl_container
from paddle.distributed.auto_parallel.dist_context import DistributedContext, DistributedOperatorContext
from paddle.distributed.auto_parallel.dist_attribute import OperatorDistributedAttribute, TensorDistributedAttribute
from paddle.distributed.auto_parallel.dist_op import DistributedOperator

paddle.enable_static()
device = "gpu" if core.is_compiled_with_cuda() else "cpu"


class MLPLayer(nn.Layer):
    def __init__(self,
                 hidden_size=1024,
                 intermediate_size=4 * 1024,
                 initializer_range=0.02):
        super(MLPLayer, self).__init__()
        d_model = hidden_size
        dim_feedforward = intermediate_size
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.Normal(
            mean=0.0, std=initializer_range))
        bias_attr = None

        self.linear0 = nn.Linear(
            d_model, dim_feedforward, weight_attr, bias_attr=bias_attr)
        self.linear1 = nn.Linear(
            dim_feedforward, d_model, weight_attr, bias_attr=bias_attr)
        self.norm = nn.LayerNorm(d_model, epsilon=1e-5)

    def forward(self, input):
        out = self.norm(input)
        out = self.linear0(out)
        out = F.gelu(out, approximate=True)
        out = self.linear1(out)

        return out


def mlp_forward(train_program, start_program):
    with static.program_guard(train_program,
                              start_program), utils.unique_name.guard():
        batch_size = 4
        hidden_size = 1024
        sqrt_hidden_size = 32
        input = static.data(
            name="input",
            shape=[sqrt_hidden_size, sqrt_hidden_size],
            dtype='int32')
        embedding = paddle.nn.Embedding(2, batch_size, sparse=True)
        input = embedding(input)
        input = paddle.reshape(input, [hidden_size, batch_size])
        input = paddle.transpose(input, perm=[1, 0])
        label = static.data(
            name="label", shape=[batch_size, 1], dtype='float32')

        mlp = MLPLayer(
            hidden_size=hidden_size,
            intermediate_size=4 * hidden_size,
            initializer_range=0.02)

        predict = mlp(input)
        error_cost = paddle.nn.functional.square_error_cost(predict, label)
        loss = paddle.mean(error_cost)
        m = paddle.nn.Softmax()
        loss = m(loss)

    return loss, train_program, start_program


class Testcompatible(unittest.TestCase):
    def test_reshape_compatible(self):
        valid_op_dist_attr_list = []
        program = paddle.static.Program()
        startup_program = paddle.static.Program()
        loss, program, start_program = mlp_forward(program, startup_program)
        ops = program.global_block().ops
        for idx, op in enumerate(ops):
            if op.type == 'reshape2':
                dist_op_impl_container = get_distributed_operator_impl_container(
                    op.type)
                op_dist_attr = OperatorDistributedAttribute()
                op_dist_attr.set_input_dims_mapping(op.input_arg_names[0],
                                                    [-1, -1, -1])
                op_dist_attr.set_output_dims_mapping(op.output_arg_names[0],
                                                     [-1, -1])
                op_dist_attr.set_output_dims_mapping(op.output_arg_names[1],
                                                     [-1, -1, -1, -1])
                dist_op = DistributedOperator(op, op_dist_attr)
                impls = dist_op_impl_container.get_impls()
                for idx, impl in enumerate(impls):
                    if idx == 1:
                        self.assertTrue(impl.is_auto_compatible(dist_op))
                    else:
                        self.assertFalse(impl.is_auto_compatible(dist_op))

    def test_transpose_compatible(self):
        valid_op_dist_attr_list = []
        program = paddle.static.Program()
        startup_program = paddle.static.Program()
        loss, program, start_program = mlp_forward(program, startup_program)
        ops = program.global_block().ops
        for idx, op in enumerate(ops):
            if op.type == 'transpose2':
                dist_op_impl_container = get_distributed_operator_impl_container(
                    op.type)
                op_dist_attr = OperatorDistributedAttribute()
                op_dist_attr.set_input_dims_mapping(op.input_arg_names[0],
                                                    [-1, -1])
                op_dist_attr.set_output_dims_mapping(op.output_arg_names[0],
                                                     [-1, -1])
                op_dist_attr.set_output_dims_mapping(op.output_arg_names[1],
                                                     [-1, -1, -1])
                dist_op = DistributedOperator(op, op_dist_attr)
                impls = dist_op_impl_container.get_impls()
                for idx, impl in enumerate(impls):
                    self.assertTrue(impl.is_auto_compatible(dist_op))

    def test_softmax_compatible(self):
        valid_op_dist_attr_list = []
        program = paddle.static.Program()
        startup_program = paddle.static.Program()
        loss, program, start_program = mlp_forward(program, startup_program)
        ops = program.global_block().ops
        for idx, op in enumerate(ops):
            if op.type == 'softmax':
                dist_op_impl_container = get_distributed_operator_impl_container(
                    op.type)
                op_dist_attr = OperatorDistributedAttribute()
                op_dist_attr.set_input_dims_mapping(op.input_arg_names[0],
                                                    [-1, -1])
                op_dist_attr.set_output_dims_mapping(op.output_arg_names[0],
                                                     [-1, -1])
                dist_op = DistributedOperator(op, op_dist_attr)
                impls = dist_op_impl_container.get_impls()
                for idx, impl in enumerate(impls):
                    self.assertTrue(impl.is_auto_compatible(dist_op))

    def test_embedding_compatible(self):
        valid_op_dist_attr_list = []
        program = paddle.static.Program()
        startup_program = paddle.static.Program()
        loss, program, start_program = mlp_forward(program, startup_program)
        ops = program.global_block().ops
        for idx, op in enumerate(ops):
            if op.type == 'embedding':
                dist_op_impl_container = get_distributed_operator_impl_container(
                    op.type)
                op_dist_attr = OperatorDistributedAttribute()
                op_dist_attr.set_input_dims_mapping(op.input_arg_names[0],
                                                    [-1, -1])
                op_dist_attr.set_output_dims_mapping(op.output_arg_names[0],
                                                     [-1, -1])
                dist_op = DistributedOperator(op, op_dist_attr)
                impls = dist_op_impl_container.get_impls()
                for idx, impl in enumerate(impls):
                    self.assertTrue(impl.is_auto_compatible(dist_op))

    def test_matmul_v2_2_compatible(self):
        valid_op_dist_attr_list = []
        program = paddle.static.Program()
        startup_program = paddle.static.Program()
        loss, program, start_program = mlp_forward(program, startup_program)
        ops = program.global_block().ops
        for idx, op in enumerate(ops):
            if op.type == 'matmul_v2':
                dist_op_impl_container = get_distributed_operator_impl_container(
                    op.type)
                op_dist_attr = OperatorDistributedAttribute()
                op_dist_attr.set_input_dims_mapping(op.input_arg_names[0],
                                                    [-1, -1])
                op_dist_attr.set_input_dims_mapping(op.input_arg_names[1],
                                                    [-1, -1])
                op_dist_attr.set_output_dims_mapping(op.output_arg_names[0],
                                                     [-1, -1])
                dist_op = DistributedOperator(op, op_dist_attr)
                impls = dist_op_impl_container.get_impls()
                for idx, impl in enumerate(impls):
                    if idx == 2:
                        self.assertTrue(impl.is_auto_compatible(dist_op))

    def test_matmul_v2_1_compatible(self):
        valid_op_dist_attr_list = []
        program = paddle.static.Program()
        startup_program = paddle.static.Program()
        loss, program, start_program = mlp_forward(program, startup_program)
        ops = program.global_block().ops
        for idx, op in enumerate(ops):
            if op.type == 'matmul_v2':
                dist_op_impl_container = get_distributed_operator_impl_container(
                    op.type)
                op_dist_attr = OperatorDistributedAttribute()
                op_dist_attr.set_input_dims_mapping(op.input_arg_names[0],
                                                    [-1, 1])
                op_dist_attr.set_input_dims_mapping(op.input_arg_names[1],
                                                    [1, -1])
                op_dist_attr.set_output_dims_mapping(op.output_arg_names[0],
                                                     [-1, -1])
                dist_op = DistributedOperator(op, op_dist_attr)
                impls = dist_op_impl_container.get_impls()
                for idx, impl in enumerate(impls):
                    if idx == 1:
                        self.assertTrue(impl.is_auto_compatible(dist_op))

    def test_matmul_v2_0_compatible(self):
        valid_op_dist_attr_list = []
        program = paddle.static.Program()
        startup_program = paddle.static.Program()
        loss, program, start_program = mlp_forward(program, startup_program)
        ops = program.global_block().ops
        for idx, op in enumerate(ops):
            if op.type == 'matmul_v2':
                dist_op_impl_container = get_distributed_operator_impl_container(
                    op.type)
                op_dist_attr = OperatorDistributedAttribute()
                op_dist_attr.set_input_dims_mapping(op.input_arg_names[0],
                                                    [-1, -1])
                op_dist_attr.set_input_dims_mapping(op.input_arg_names[1],
                                                    [-1, 1])
                op_dist_attr.set_output_dims_mapping(op.output_arg_names[0],
                                                     [-1, 1])
                dist_op = DistributedOperator(op, op_dist_attr)
                impls = dist_op_impl_container.get_impls()
                for idx, impl in enumerate(impls):
                    if idx == 0:
                        self.assertTrue(impl.is_auto_compatible(dist_op))


if __name__ == "__main__":
    unittest.main()
