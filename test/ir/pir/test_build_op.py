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
            y_s = paddle.tanh(y_s)
        pir_program = pir.translate_to_pir(main_program.desc)
        return pir_program


class TestBuildOp(unittest.TestCase):
    def test_build_mean_op(self):
        pir_program = get_ir_program()
        tanh_out = pir_program.global_block().ops[-1].result(0)
        with paddle.pir_utils.IrGuard(), paddle.pir.core.program_guard(
            pir_program
        ):
            out = paddle.mean(tanh_out)
            self.assertEqual(out.get_defining_op().name(), "pd_op.mean")
            self.assertEqual(
                out.get_defining_op()
                .operands()[0]
                .source()
                .get_defining_op()
                .name(),
                "pd_op.tanh",
            )
        paddle.pir.create_shaped_type(tanh_out.type(), [3148873728])
        paddle.pir.create_shaped_type(tanh_out.type(), [1])


class TestBuildOp2(unittest.TestCase):
    def test_build_add_n_op(self):
        pir_program = get_ir_program()
        tanh_out = pir_program.global_block().ops[-1].result(0)
        with paddle.pir_utils.IrGuard(), paddle.pir.core.program_guard(
            pir_program
        ):
            out1 = paddle.mean(tanh_out)
            out2 = paddle.mean(tanh_out)
            out = paddle.add_n([out1, out2])
            self.assertEqual(out.get_defining_op().name(), "pd_op.add_n")
            self.assertEqual(
                out.get_defining_op()
                .operands()[0]
                .source()
                .get_defining_op()
                .name(),
                "builtin.combine",
            )


class TestBuildOp3(unittest.TestCase):
    def test_insertion_point(self):
        pir_program = get_ir_program()
        with paddle.pir_utils.IrGuard():
            add_op = pir_program.global_block().ops[-2]
            tanh_op = pir_program.global_block().ops[-1]
            add_out = add_op.result(0)
            tanh_operand = tanh_op.operands()[0]

            with paddle.pir.core.program_guard(pir_program):
                pir.set_insertion_point(tanh_op)
                full_out = paddle.tensor.fill_constant(
                    shape=[4, 4], dtype="float", value=2
                )
                divide_out = paddle.divide(full_out, full_out)
                sum_out = paddle.sum(divide_out)
                out = paddle.mean(sum_out)
                tanh_operand.set_source(out)

            print(pir_program)
            self.assertEqual(
                tanh_operand.source().get_defining_op().name(), "pd_op.mean"
            )


class TestBuildOp4(unittest.TestCase):
    def test_build_concat_op(self):
        pir_program = get_ir_program()
        tanh_out = pir_program.global_block().ops[-1].result(0)
        with paddle.pir_utils.IrGuard(), paddle.pir.core.program_guard(
            pir_program
        ):
            out = paddle.concat([tanh_out, tanh_out], 0)
            self.assertEqual(out.get_defining_op().name(), "pd_op.concat")
            self.assertEqual(
                out.get_defining_op()
                .operands()[0]
                .source()
                .get_defining_op()
                .name(),
                "builtin.combine",
            )


class TestBuildOp5(unittest.TestCase):
    def test_build_split_op(self):
        pir_program = get_ir_program()
        tanh_out = pir_program.global_block().ops[-1].result(0)
        with paddle.pir_utils.IrGuard(), paddle.pir.core.program_guard(
            pir_program
        ):
            out = paddle.split(tanh_out, [2, 2], 0)
            self.assertEqual(out[0].get_defining_op().name(), "builtin.split")
            self.assertEqual(
                out[0]
                .get_defining_op()
                .operands()[0]
                .source()
                .get_defining_op()
                .name(),
                "pd_op.split",
            )


class TestBuildOp6(unittest.TestCase):
    def test_build_tensorrt_engine_op(self):
        pir_program = get_ir_program()
        tanh_out = pir_program.global_block().ops[-1].result(0)
        with paddle.pir_utils.IrGuard(), paddle.pir.core.program_guard(
            pir_program
        ):
            # create fake tensorrt op
            trt_params = paddle.base.libpaddle.TRTEngineParams()
            trt_params.min_input_shape = {"x": [1, 1]}
            trt_params.max_input_shape = {"x": [10, 1]}
            trt_params.optim_input_shape = {"x": [5, 1]}
            trt_params.engine_serialized_data = ""
            out = paddle._C_ops.tensorrt_engine(
                [tanh_out],
                trt_params,
                ["x"],
                ["out"],
                [[1, 1]],
                [paddle.base.libpaddle.DataType.FLOAT32],
                "NO DEBUG",
            )
            self.assertEqual(
                out[0]
                .get_defining_op()
                .operands()[0]
                .source()
                .get_defining_op()
                .name(),
                "pd_op.tensorrt_engine",
            )


class TestGetValueByOpId(unittest.TestCase):
    def test_get_value_by_op_id(self):
        def true_func():
            return paddle.tensor.fill_constant(
                shape=[2, 3], dtype='int32', value=2
            )

        def false_func():
            return paddle.tensor.fill_constant(
                shape=[3, 2], dtype='int32', value=-1
            )

        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.tensor.fill_constant(
                shape=[1], dtype='float32', value=0.1
            )
            y = paddle.tensor.fill_constant(
                shape=[1], dtype='float32', value=0.23
            )
            pred = paddle.less_than(y, x)
            out = paddle.static.nn.cond(pred, true_func, false_func)
            value1 = main_program.get_value_by_op_id(69)
            self.assertEqual(
                out.get_defining_op().id(),
                value1[0].get_defining_op().id(),
            )
            value2 = main_program.get_value_by_op_id([58, 69])
            self.assertEqual(
                69,
                value2[0].get_defining_op().id(),
            )


if __name__ == "__main__":
    unittest.main()
