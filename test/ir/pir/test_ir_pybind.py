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
from paddle.autograd.backward_utils import ValueSet

paddle.enable_static()


def get_ir_program():
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
            z_s = paddle.add(y_s, y_s)
            k_s = paddle.tanh(z_s)
            q_s = paddle.unsqueeze(k_s, [2])

        pir_program = pir.translate_to_pir(main_program.desc)
        return pir_program


class TestPybind(unittest.TestCase):
    def test_program(self):
        pir_program = get_ir_program()

        block = pir_program.global_block()
        program = block.program

        self.assertEqual(pir_program, program)

        self.assertEqual(len(pir_program.blocks), 1)

    def test_block(self):
        pir_program = get_ir_program()
        block = pir_program.global_block()
        print(block)  # test print block
        ops = block.ops
        self.assertEqual(
            len(ops), 6
        )  # pir program add "builtin.parameter" by default, so size is 4
        block.remove_op(ops[5])
        self.assertEqual(len(block.ops), 5)

    def test_operation(self):
        pir_program = get_ir_program()
        ops = pir_program.global_block().ops
        matmul_op = ops[1]
        add_op = ops[2]
        tanh_op = ops[3]
        parent_block = tanh_op.get_parent_block()
        parent_ops_num = len(parent_block.ops)
        self.assertEqual(parent_ops_num, 6)
        self.assertEqual(tanh_op.num_results(), 1)
        self.assertEqual(len(matmul_op.get_input_names()), 2)
        self.assertEqual(len(matmul_op.get_attr_names()), 2)
        self.assertEqual(len(matmul_op.get_output_names()), 1)
        # test operand.index
        self.assertEqual(matmul_op.operand(0).index(), 0)
        self.assertEqual(matmul_op.operand(1).index(), 1)
        self.assertEqual(add_op.operand(0).index(), 0)
        self.assertEqual(add_op.operand(1).index(), 1)
        self.assertEqual(tanh_op.operand(0).index(), 0)

    def test_value(self):
        pir_program = get_ir_program()
        matmul_op = pir_program.global_block().ops[1]
        add_op = pir_program.global_block().ops[2]
        tanh_op = pir_program.global_block().ops[3]

        self.assertEqual(
            matmul_op.result(0).dtype, paddle.base.core.DataType.FLOAT32
        )
        self.assertEqual(matmul_op.result(0).shape, [4, 4])
        self.assertEqual(
            matmul_op.results()[0].get_defining_op().name(), "pd_op.matmul"
        )
        self.assertEqual(
            matmul_op.result(0).get_defining_op().name(), "pd_op.matmul"
        )
        matmul_op.result(0).stop_gradient = True
        self.assertEqual(matmul_op.result(0).stop_gradient, True)

        # test opresult hash
        result_set = ValueSet()
        for opresult in matmul_op.results():
            result_set.add(opresult)
        # test opresult hash and hash(opresult) == hash(operesult)
        self.assertTrue(add_op.operands()[0].source() in result_set)
        # test value hash and hash(value) == hash(operesult)
        self.assertTrue(add_op.operands_source()[0] in result_set)
        # test value == value
        self.assertTrue(
            add_op.operands_source()[0].is_same(add_op.operands_source()[0])
        )
        # test value == opresult
        self.assertTrue(
            add_op.operands_source()[0].is_same(matmul_op.results()[0])
        )
        # test opresult print
        self.assertTrue(
            'dtype=tensor<4x4xf32>' in add_op.operands_source()[0].__str__()
        )
        # test opresult == value
        self.assertTrue(
            add_op.operands()[0].source().is_same(add_op.operands_source()[0])
        )
        # test opresult == opresult
        self.assertTrue(
            add_op.operands()[0].source().is_same(matmul_op.results()[0])
        )

        # test opresult print
        self.assertEqual(
            tanh_op.operands()[0].source().get_defining_op().name(), "pd_op.add"
        )
        self.assertTrue(
            'tensor<4x4xf32>' in tanh_op.operands()[0].source().__str__()
        )
        add_op.replace_all_uses_with(matmul_op.results())
        self.assertEqual(
            tanh_op.operands()[0].source().get_defining_op().name(),
            "pd_op.matmul",
        )

        self.assertEqual(add_op.result(0).use_empty(), True)

        self.assertEqual(add_op.result(0).initialized(), True)

        uninit_value = paddle.pir.Value()
        self.assertEqual(uninit_value.initialized(), False)

    def test_type(self):
        pir_program = get_ir_program()
        matmul_op = pir_program.global_block().ops[1]
        add_op = pir_program.global_block().ops[2]
        self.assertEqual(
            matmul_op.result(0).type() == add_op.result(0).type(), True
        )
        add_op.result(0).set_type(
            paddle.base.libpaddle.pir.create_selected_rows_type_by_dense_tensor(
                add_op.result(0).type()
            )
        )
        self.assertEqual(add_op.result(0).is_selected_row_type(), True)

    def test_attr(self):
        with paddle.pir_utils.OldIrGuard():
            main_program, start_program = (
                paddle.static.Program(),
                paddle.static.Program(),
            )
            with paddle.static.program_guard(main_program, start_program):
                conv_data = paddle.static.data(
                    'conv_data', [None, 3, 32, 32], dtype='float32'
                )
                conv2d_out = paddle.static.nn.conv2d(
                    input=conv_data,
                    num_filters=2,
                    filter_size=3,
                    stride=3,
                    act="relu",
                )
                full_out = paddle.tensor.fill_constant(
                    shape=[4, 4], dtype="float32", value=2
                )

            pir_program = pir.translate_to_pir(main_program.desc)
            conv_attr = pir_program.global_block().ops[3].attrs()
            full_attr = pir_program.global_block().ops[8].attrs()
            self.assertEqual(conv_attr["stop_gradient"], [False])
            self.assertEqual(conv_attr["dilations"], [1, 1])
            self.assertEqual(conv_attr["data_format"], "NCHW")
            self.assertEqual(conv_attr["strides"], [3, 3])
            self.assertEqual(conv_attr["paddings"], [0, 0])
            self.assertEqual(conv_attr["padding_algorithm"], "EXPLICIT")
            self.assertEqual(conv_attr["groups"], 1)
            self.assertEqual(
                full_attr["dtype"], paddle.base.core.DataType.FLOAT32
            )
            self.assertTrue(
                isinstance(full_attr["place"], paddle.base.core.Place)
            )

    def test_operands(self):
        pir_program = get_ir_program()
        matmul_op = pir_program.global_block().ops[1]
        operands = matmul_op.operands()
        self.assertEqual(len(operands), 2)

    def test_results(self):
        pir_program = get_ir_program()
        matmul_op = pir_program.global_block().ops[1]
        results = matmul_op.results()
        self.assertEqual(len(results), 1)

    def test_get_output_intermediate_status(self):
        pir_program = get_ir_program()
        unsqueeze_op = pir_program.global_block().ops[-1]
        results = unsqueeze_op.get_output_intermediate_status()
        self.assertEqual(results, [False])

    def test_prog_seed(self):
        p = pir.Program()
        self.assertEqual(p._seed, 0)

        p.global_seed(10)
        self.assertEqual(p._seed, 10)

    def test_opresult_id(self):
        with paddle.pir_utils.IrGuard():
            a = paddle.static.data(name='a', shape=[4, 4], dtype='float32')
            result = paddle.tanh(a)

        self.assertIsInstance(a.id, str)
        self.assertIsInstance(result.id, str)

    def test_operation_get_input_names_error(self):
        """It will Raise error if operation `builtin.set_parameter` calls `get_input_names`. Because `builtin.set_parameter` does not have OpYamlInfoInterface"""
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            startup = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                param1 = paddle.pir.core.create_parameter(
                    dtype="float32",
                    shape=[5, 10],
                    name="param1",
                    initializer=paddle.nn.initializer.Uniform(),
                )

                block = startup.global_block()
                set_parameter_ops = [
                    op
                    for op in block.ops
                    if op.name() == 'builtin.set_parameter'
                ]
                set_parameter_op = set_parameter_ops[0]
                parameter_name = set_parameter_op.attrs()["parameter_name"]
                with self.assertRaises(ValueError):
                    input_names = set_parameter_op.get_input_names()


if __name__ == "__main__":
    unittest.main()
