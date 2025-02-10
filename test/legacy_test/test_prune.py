#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

paddle.enable_static()


class TestPruneBase(unittest.TestCase):
    def run_net(self, net):
        program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(program, startup_program):
            ret = net()

        return ret, program

    def check_prune_with_input(
        self,
        program,
        feeded_vars,
        targets,
        ops_before_pruned,
        ops_after_pruned,
    ):
        block = program.global_block()
        self.assertEqual(len(block.ops), len(ops_before_pruned))
        self.assertEqual(
            [op.name() for op in block.ops],
            ops_before_pruned,
        )
        pruned_program = program._prune_with_input(
            feeded_vars=feeded_vars, targets=targets
        )
        self.assertEqual(
            len(pruned_program.global_block().ops), len(ops_after_pruned)
        )
        self.assertEqual(
            [op.name() for op in pruned_program.global_block().ops],
            ops_after_pruned,
        )

    def check_prune(
        self, program, targets, ops_before_pruned, ops_after_pruned
    ):
        block = program.global_block()
        self.assertEqual(len(block.ops), len(ops_before_pruned))
        self.assertEqual(
            [op.name() for op in block.ops],
            ops_before_pruned,
        )
        pruned_program = program._prune(targets=targets)
        self.assertEqual(
            len(pruned_program.global_block().ops), len(ops_after_pruned)
        )
        self.assertEqual(
            [op.name() for op in pruned_program.global_block().ops],
            ops_after_pruned,
        )

    def check_prune_target_not_list(
        self, program, targets, ops_before_pruned, ops_after_pruned
    ):
        block = program.global_block()
        self.assertEqual(len(block.ops), len(ops_before_pruned))
        self.assertEqual(
            [op.name() for op in block.ops],
            ops_before_pruned,
        )
        pruned_program = program._prune(targets=targets)
        self.assertEqual(
            len(pruned_program.global_block().ops), len(ops_after_pruned)
        )
        self.assertEqual(
            [op.name() for op in pruned_program.global_block().ops],
            ops_after_pruned,
        )

    def check_prune_target_none(self, program, ops_before_pruned):
        block = program.global_block()
        self.assertEqual(len(block.ops), len(ops_before_pruned))
        self.assertEqual(
            [op.name() for op in block.ops],
            ops_before_pruned,
        )
        try:
            pruned_program = program._prune(targets=None)
        except TypeError as e:
            self.assertIn(
                "_prune(): incompatible function arguments. The following argument types are supported:",
                str(e),
            )


class TestPrune(TestPruneBase):
    def net(self):
        x = paddle.static.data(name='x', shape=[-1, 2], dtype='float32')
        label = paddle.static.data(name="label", shape=[-1, 1], dtype="int64")
        y = paddle.static.nn.fc(x=[x], size=2, activation="softmax")
        loss = paddle.nn.functional.cross_entropy(
            input=y, label=label, reduction='none', use_softmax=False
        )
        loss = paddle.mean(x=loss)
        return x, y, label, loss

    def test_prune_with_input(self):
        ops_before_pruned = [
            "builtin.parameter",
            "builtin.parameter",
            "pd_op.data",
            "pd_op.data",
            "pd_op.matmul",
            "pd_op.add",
            "pd_op.softmax",
            "pd_op.cross_entropy_with_softmax",
            "pd_op.full_int_array",
            "pd_op.mean",
        ]

        ops_after_pruned = [
            "pd_op.data",
            "pd_op.data",
            "pd_op.cross_entropy_with_softmax",
            "pd_op.full_int_array",
            "pd_op.mean",
        ]
        (x, y, label, loss), program = self.run_net(self.net)
        self.check_prune_with_input(
            program,
            [y, label],
            [loss],
            ops_before_pruned,
            ops_after_pruned,
        )

    def test_prune(self):
        ops_before_pruned = [
            "builtin.parameter",
            "builtin.parameter",
            "pd_op.data",
            "pd_op.data",
            "pd_op.matmul",
            "pd_op.add",
            "pd_op.softmax",
            "pd_op.cross_entropy_with_softmax",
            "pd_op.full_int_array",
            "pd_op.mean",
        ]

        ops_after_pruned = [
            "builtin.parameter",
            "builtin.parameter",
            "pd_op.data",
            "pd_op.data",
            "pd_op.matmul",
            "pd_op.add",
            "pd_op.softmax",
            "pd_op.cross_entropy_with_softmax",
            "pd_op.full_int_array",
            "pd_op.mean",
        ]

        (x, y, label, loss), program = self.run_net(self.net)
        self.check_prune(program, [loss], ops_before_pruned, ops_after_pruned)

    def test_prune_target_not_list(self):
        ops_before_pruned = [
            "builtin.parameter",
            "builtin.parameter",
            "pd_op.data",
            "pd_op.data",
            "pd_op.matmul",
            "pd_op.add",
            "pd_op.softmax",
            "pd_op.cross_entropy_with_softmax",
            "pd_op.full_int_array",
            "pd_op.mean",
        ]

        ops_after_pruned = [
            "builtin.parameter",
            "builtin.parameter",
            "pd_op.data",
            "pd_op.data",
            "pd_op.matmul",
            "pd_op.add",
            "pd_op.softmax",
            "pd_op.cross_entropy_with_softmax",
            "pd_op.full_int_array",
            "pd_op.mean",
        ]

        (x, y, label, loss), program = self.run_net(self.net)

        self.check_prune_target_not_list(
            program, [loss], ops_before_pruned, ops_after_pruned
        )

    def test_prune_target_none(self):
        ops_before_pruned = [
            "builtin.parameter",
            "builtin.parameter",
            "pd_op.data",
            "pd_op.data",
            "pd_op.matmul",
            "pd_op.add",
            "pd_op.softmax",
            "pd_op.cross_entropy_with_softmax",
            "pd_op.full_int_array",
            "pd_op.mean",
        ]

        (x, y, label, loss), program = self.run_net(self.net)
        self.check_prune_target_none(program, ops_before_pruned)


if __name__ == '__main__':
    unittest.main()
