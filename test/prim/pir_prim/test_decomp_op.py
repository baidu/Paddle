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

import unittest

import paddle
from paddle import pir
from paddle.decomposition import decompose
from paddle.framework import core

paddle.enable_static()


def get_ir_program():
    paddle.enable_static()
    with paddle.pir_utils.OldIrGuard():
        x = paddle.randn([4, 4])
        main_program, start_program = (
            paddle.static.Program(),
            paddle.static.Program(),
        )
        with paddle.static.program_guard(main_program, start_program):
            x_s = paddle.static.data('x', [4, 4], x.dtype)
            x_s.stop_gradient = False
            y_s = paddle.matmul(x_s, x_s)
            y_s = paddle.add(x_s, y_s)
            y_s = paddle.mean(y_s)
            y_s = paddle.tanh(y_s)
        pir_program = pir.translate_to_pir(main_program.desc)

        all_ops = pir_program.global_block().ops
        for op in all_ops:
            op.op_role = 1

        return pir_program


class TestBuildOp(unittest.TestCase):
    def test_build_op(self):
        pir_program = get_ir_program()
        y = pir_program.global_block().ops[-2].results()
        orig_shape = y[0].shape
        with paddle.pir_utils.IrGuard():
            core._set_prim_forward_enabled(True)
            y_new = decompose(pir_program, y)
            core._set_prim_forward_enabled(False)
            new_shape = y_new[0].shape
            assert (
                orig_shape == new_shape
            ), f"Original shape {orig_shape} is not equal to new shape {new_shape}"
            op_name_list = [op.name() for op in pir_program.global_block().ops]
            self.assertEqual(
                op_name_list,
                [
                    'pd_op.data',
                    'pd_op.matmul',
                    'pd_op.add',
                    'pd_op.full_int_array',
                    'pd_op.full_int_array',
                    'pd_op.sum',
                    'pd_op.full',
                    'pd_op.divide',
                    'pd_op.tanh',
                ],
            )
            op_role_list = [op.op_role for op in pir_program.global_block().ops]
            self.assertEqual(
                all(op_role == 1 for op_role in op_role_list), True
            )


if __name__ == "__main__":
    unittest.main()
