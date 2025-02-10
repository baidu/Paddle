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
from paddle.base.core import call_vjp, has_vjp

paddle.enable_static()


def get_ir_program():
    with paddle.pir_utils.OldIrGuard():
        main_program, start_program = (
            paddle.static.Program(),
            paddle.static.Program(),
        )
        with paddle.static.program_guard(main_program, start_program):
            x = paddle.static.data('x', [4, 4], 'float32')
            x.stop_gradient = False
            paddle.tanh(x)
            paddle.tensor.fill_constant(
                shape=[4, 4], dtype='float32', value=2.0
            )
        pir_program = pir.translate_to_pir(main_program.desc)
        return pir_program


class TestTanhVjp(unittest.TestCase):
    def test_tanh_vjp1(self):
        pir_program = get_ir_program()
        tanh_op = pir_program.global_block().ops[-2]
        fill_constant_op = pir_program.global_block().ops[-1]
        out_grads = [[fill_constant_op.result(0)]]
        stop_gradients = [[False]]
        with paddle.pir.core.program_guard(pir_program):
            grad_outs = call_vjp(
                tanh_op,
                [[tanh_op.operand_source(0)]],
                [[tanh_op.result(0)]],
                out_grads,
                stop_gradients,
            )
        self.assertEqual(
            grad_outs[0][0].get_defining_op().name(), "pd_op.tanh_grad"
        )
        self.assertEqual(
            grad_outs[0][0]
            .get_defining_op()
            .operands()[0]
            .source()
            .get_defining_op()
            .name(),
            "pd_op.tanh",
        )
        self.assertEqual(
            grad_outs[0][0]
            .get_defining_op()
            .operands()[1]
            .source()
            .get_defining_op()
            .name(),
            "pd_op.full",
        )
        self.assertEqual(len(pir_program.global_block().ops), 4)

    def test_tanh_vjp2(self):
        pir_program = get_ir_program()
        tanh_op = pir_program.global_block().ops[-2]
        fill_constant_op = pir_program.global_block().ops[-1]
        out_grads = [[fill_constant_op.result(0)]]
        stop_gradients = [[True]]
        with paddle.pir.core.program_guard(pir_program):
            grad_outs = call_vjp(
                tanh_op,
                [[tanh_op.operand_source(0)]],
                [[tanh_op.result(0)]],
                out_grads,
                stop_gradients,
            )
        self.assertEqual(grad_outs[0][0], None)


class TestMeanVjp(unittest.TestCase):
    def test_mean_vjp1(self):
        with paddle.pir_utils.OldIrGuard():
            main_program, start_program = (
                paddle.static.Program(),
                paddle.static.Program(),
            )
            with paddle.static.program_guard(main_program, start_program):
                x = paddle.static.data('x', [4, 4], 'float32')
                x.stop_gradient = False
                paddle.mean(x, axis=[0, 1])
                paddle.tensor.fill_constant(
                    shape=[1], dtype='float32', value=2.0
                )
            pir_program = pir.translate_to_pir(main_program.desc)
            fill_constant_op = pir_program.global_block().ops[-1]
            mean_op = pir_program.global_block().ops[-2]
            out_grads = [[fill_constant_op.result(0)]]
            stop_gradients = [[False]]
        with paddle.pir.core.program_guard(pir_program):
            grad_outs = call_vjp(
                mean_op,
                [[mean_op.operand_source(0)], [mean_op.operand_source(1)]],
                [[mean_op.result(0)]],
                out_grads,
                stop_gradients,
            )
            self.assertEqual(
                grad_outs[0][0].get_defining_op().name(), "pd_op.mean_grad"
            )
            self.assertEqual(
                grad_outs[0][0]
                .get_defining_op()
                .operands()[0]
                .source()
                .get_defining_op()
                .name(),
                "pd_op.data",
            )
            self.assertEqual(
                grad_outs[0][0]
                .get_defining_op()
                .operands()[1]
                .source()
                .get_defining_op()
                .name(),
                "pd_op.full",
            )
            self.assertEqual(len(pir_program.global_block().ops), 5)

    def test_mean_vjp2(self):
        with paddle.pir_utils.OldIrGuard():
            main_program, start_program = (
                paddle.static.Program(),
                paddle.static.Program(),
            )
            with paddle.static.program_guard(main_program, start_program):
                x = paddle.static.data('x', [4, 4], 'float32')
                x.stop_gradient = False
                paddle.mean(x, axis=[0, 1])
                paddle.tensor.fill_constant(
                    shape=[1], dtype='float32', value=2.0
                )
            pir_program = pir.translate_to_pir(main_program.desc)
            fill_constant_op = pir_program.global_block().ops[-1]
            mean_op = pir_program.global_block().ops[-2]
            out_grads = [[fill_constant_op.result(0)]]
            stop_gradients = [[True]]
        with paddle.pir.core.program_guard(pir_program):
            grad_outs = call_vjp(
                mean_op,
                [[mean_op.operand_source(0)], [mean_op.operand_source(1)]],
                [[mean_op.result(0)]],
                out_grads,
                stop_gradients,
            )
            self.assertEqual(grad_outs[0][0], None)


class TesthasVjp(unittest.TestCase):
    def test_has_vjp(self):
        with paddle.pir_utils.OldIrGuard():
            main_program, start_program = (
                paddle.static.Program(),
                paddle.static.Program(),
            )
            with paddle.static.program_guard(main_program, start_program):
                x = paddle.static.data('x', [4, 4], 'float32')
                x.stop_gradient = False
                paddle.mean(x, axis=[0, 1])
                paddle.tensor.fill_constant(
                    shape=[1], dtype='float32', value=2.0
                )
            pir_program = pir.translate_to_pir(main_program.desc)
            fill_constant_op = pir_program.global_block().ops[-1]
            mean_op = pir_program.global_block().ops[-2]
            self.assertEqual(has_vjp(fill_constant_op), False)
            self.assertEqual(has_vjp(mean_op), True)


if __name__ == "__main__":
    unittest.main()
