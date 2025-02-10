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
from paddle.autograd.ir_backward import grad
from paddle.base.core import call_vjp, has_vjp
from paddle.base.libpaddle.pir import (
    build_pipe_for_block,
    get_used_external_value,
)

paddle.enable_static()


class TestConstructModuleWithPyLayerOp(unittest.TestCase):
    def test_fwd_only_with_single_output(self):
        """
        pseudocode:

        y = 3 * x
        """

        def forward_fn(x):
            return 3 * x

        with paddle.pir_utils.IrGuard():
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            with paddle.static.program_guard(main_program, startup_program):
                x = paddle.static.data(name="x", shape=[6, 1], dtype="float32")
                x.stop_gradient = True
                out = paddle.static.nn.static_pylayer(
                    forward_fn, [x], backward_fn=None
                )
                y = paddle.mean(out)

            pylayer_op = main_program.global_block().ops[-3]
            self.assertEqual(pylayer_op.name(), "pd_op.pylayer")
            self.assertEqual(len(pylayer_op.results()), 1)
            value_list = get_used_external_value(pylayer_op)
            self.assertEqual(len(value_list), 1)
            self.assertTrue(value_list[0].is_same(pylayer_op.operand_source(0)))

    def test_fwd_only_with_multi_inputs_multi_outputs(self):
        """
        pseudocode:

        ret1 = x * y
        ret2 = x - y
        """

        def forward_fn(x, y):
            z = 3 * x
            return z * y, z - y

        with paddle.pir_utils.IrGuard():
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            with paddle.static.program_guard(main_program, startup_program):
                x = paddle.static.data(name="x", shape=[6, 1], dtype="float32")
                y = paddle.static.data(name="x", shape=[6, 1], dtype="float32")
                x.stop_gradient = True
                y.stop_gradient = True
                ret1, ret2 = paddle.static.nn.static_pylayer(
                    forward_fn, [x, y], backward_fn=None
                )
                out = ret1 + ret2

            pylayer_op = main_program.global_block().ops[-2]
            self.assertEqual(pylayer_op.name(), "pd_op.pylayer")
            self.assertEqual(len(pylayer_op.results()), 2)
            value_list = get_used_external_value(pylayer_op)
            self.assertEqual(len(value_list), 2)
            self.assertTrue(value_list[0].is_same(pylayer_op.operand_source(0)))
            self.assertTrue(value_list[1].is_same(pylayer_op.operand_source(1)))

            # check build_pipe_for_block interface
            fwd_block = pylayer_op.as_pylayer_op().forward_block()
            build_pipe_for_block(fwd_block)

    def test_pylayer_op_call_vjp_interface(self):
        def forward_fn(x, y):
            z = 3 * x
            tmp = paddle.mean(x)
            return z * y, z - y, tmp

        def backward_fn(dx, dy, dtmp):
            return dx - dy, paddle.broadcast_to(dtmp, dx.shape)

        with paddle.pir_utils.IrGuard():
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            with paddle.static.program_guard(main_program, startup_program):
                x = paddle.static.data(name="x", shape=[6, 1], dtype="float32")
                x.stop_gradient = False
                y = paddle.static.data(name="y", shape=[6, 1], dtype="float32")
                y.stop_gradient = False

                out_0, out_1, out_2 = paddle.static.nn.static_pylayer(
                    forward_fn, [x, y], backward_fn=backward_fn
                )
                ret = paddle.mean(out_0 - 2 * out_1 + out_2)

                dataop = main_program.global_block().ops[0]
                pylayer_op = main_program.global_block().ops[-7]
                self.assertEqual(pylayer_op.name(), "pd_op.pylayer")
                self.assertEqual(len(pylayer_op.results()), 3)

                out_grad_0 = paddle.full(
                    shape=[6, 1], dtype='float32', fill_value=3
                )
                out_grad_1 = paddle.full(
                    shape=[6, 1], dtype='float32', fill_value=2
                )
                out_grad_2 = paddle.full(
                    shape=[], dtype='float32', fill_value=0.1
                )
                pylayer_input = [
                    [input] for input in get_used_external_value(pylayer_op)
                ]
                pylayer_input_stop_gradients = [[False], [False]]
                pylayer_output = [pylayer_op.results()]
                pylayer_output_grad = [[out_grad_0], [out_grad_1], [out_grad_2]]
                self.assertEqual(has_vjp(pylayer_op), False)

                grad_outs = call_vjp(
                    pylayer_op,
                    pylayer_input,
                    pylayer_output,
                    pylayer_output_grad,
                    pylayer_input_stop_gradients,
                )

    def test_pylayer_op_backward(self):
        def forward_fn(x, y):
            z = 3 * x
            tmp = paddle.mean(x)
            return z * y, z - y, tmp

        def backward_fn(dx, dy, dtmp):
            return dx - dy, dtmp * dy * dx

        with paddle.pir_utils.IrGuard():
            main_program = paddle.static.Program()
            startup_program = paddle.static.Program()
            with paddle.static.program_guard(main_program, startup_program):
                x = paddle.static.data(name="x", shape=[6, 1], dtype="float32")
                x.stop_gradient = False
                y = paddle.static.data(name="y", shape=[6, 1], dtype="float32")
                y.stop_gradient = False

                out_0, out_1, out_2 = paddle.static.nn.static_pylayer(
                    forward_fn, [x, y], backward_fn=backward_fn
                )
                ret = paddle.mean(out_0 - 2 * out_1 + out_2)

                dataop = main_program.global_block().ops[0]
                pylayer_op = main_program.global_block().ops[-7]
                self.assertEqual(pylayer_op.name(), "pd_op.pylayer")
                self.assertEqual(len(pylayer_op.results()), 3)

                self.assertEqual(has_vjp(pylayer_op), False)
                dataop_0 = main_program.global_block().ops[0]
                dataop_1 = main_program.global_block().ops[1]

                # check vjp interface for if_op
                grad_outs = grad(
                    pylayer_op.results(),
                    [dataop_0.result(0), dataop_1.result(0)],
                )


if __name__ == "__main__":
    unittest.main()
