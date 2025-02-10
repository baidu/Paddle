#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import math
import unittest

import numpy as np
from scipy import special
from utils import dygraph_guard, static_guard

import paddle
from paddle.base import framework
from paddle.base.core import VarDesc
from paddle.regularizer import L2Decay

DELTA = 0.00001


def check_cast_op(op):
    return (
        op.type == 'cast'
        and op.attr('in_dtype') == VarDesc.VarType.FP32
        and op.attr('out_dtype') in [VarDesc.VarType.FP16, VarDesc.VarType.BF16]
    )


def check_cast_op_pir(op):
    return (
        op.name() == 'pd_op.cast'
        and op.attrs()['dtype']
        in (
            paddle.base.libpaddle.DataType.FLOAT16,
            paddle.base.libpaddle.DataType.BFLOAT16,
        )
        and op.operand_source(0).dtype == paddle.base.libpaddle.DataType.FLOAT32
    )


def output_hist(out):
    hist, _ = np.histogram(out, range=(-1, 1))
    hist = hist.astype("float32")
    hist /= float(out.size)
    prob = 0.1 * np.ones(10)
    return hist, prob


class TestConstantInitializer(unittest.TestCase):
    def test_calculate_gain(self):
        self.assertEqual(paddle.nn.initializer.calculate_gain('sigmoid'), 1)
        self.assertEqual(paddle.nn.initializer.calculate_gain('linear'), 1)
        self.assertEqual(paddle.nn.initializer.calculate_gain('conv2d'), 1)
        self.assertEqual(paddle.nn.initializer.calculate_gain('tanh'), 5.0 / 3)
        self.assertEqual(
            paddle.nn.initializer.calculate_gain('relu'), math.sqrt(2.0)
        )
        self.assertEqual(
            paddle.nn.initializer.calculate_gain('leaky_relu', 1), 1
        )
        self.assertEqual(paddle.nn.initializer.calculate_gain('selu'), 3.0 / 4)

    def test_constant_initializer_default_value(self, dtype="float32"):
        """Test the constant initializer with default value"""
        with paddle.pir_utils.OldIrGuard():
            program = framework.Program()
            block = program.global_block()
            for _ in range(2):
                block.create_parameter(
                    dtype=dtype,
                    shape=[5, 10],
                    name="param",
                    initializer=paddle.nn.initializer.Constant(),
                )
            num_ops = 1
            self.assertEqual(len(block.ops), num_ops)
            init_op = block.ops[0]
            self.assertEqual(init_op.type, 'fill_constant')
            self.assertAlmostEqual(init_op.attr('value'), 0.0, delta=DELTA)
            return block

    def test_constant_initializer(self, dtype="float32"):
        """Test constant initializer with supplied value"""
        with paddle.pir_utils.OldIrGuard():
            program = framework.Program()
            block = program.global_block()
            for _ in range(2):
                block.create_parameter(
                    dtype=dtype,
                    shape=[5, 10],
                    name="param",
                    initializer=paddle.nn.initializer.Constant(2.3),
                )
            num_ops = 1
            self.assertEqual(len(block.ops), num_ops)
            init_op = block.ops[0]
            self.assertEqual(init_op.type, 'fill_constant')
            self.assertAlmostEqual(init_op.attr('value'), 2.3, delta=DELTA)
            return block

    def test_constant_initializer_fp16(self):
        """Test constant initializer with float16"""
        self.test_constant_initializer_default_value("float16")
        self.test_constant_initializer("float16")

    def test_constant_initializer_bf16(self):
        """Test constant initializer with bfloat16
        No cast operator has been added here
        """
        self.test_constant_initializer_default_value("uint16")
        self.test_constant_initializer("uint16")


class TestUniformInitializer(unittest.TestCase):
    def test_uniform_initializer_default_value(self, dtype="float32"):
        """Test the uniform initializer with default value"""
        with paddle.pir_utils.OldIrGuard():
            program = framework.Program()
            block = program.global_block()
            for _ in range(2):
                block.create_parameter(
                    dtype=dtype,
                    shape=[5, 10],
                    name="param",
                    initializer=paddle.nn.initializer.Uniform(),
                )
            num_ops = 2 if dtype == "float16" else 1
            self.assertEqual(len(block.ops), num_ops)
            init_op = block.ops[0]
            self.assertEqual(init_op.type, 'uniform_random')
            self.assertAlmostEqual(init_op.attr('min'), -1.0, delta=DELTA)
            self.assertAlmostEqual(init_op.attr('max'), 1.0, delta=DELTA)
            self.assertEqual(init_op.attr('seed'), 0)
            return block

    def test_uniform_initializer_random_seed(self):
        """Test the uniform initializer with manually setting seed"""
        with paddle.pir_utils.OldIrGuard():
            program = framework.Program()
            program.random_seed = 123
            block = program.global_block()
            for _ in range(2):
                block.create_parameter(
                    dtype="float32",
                    shape=[5, 10],
                    name="param1",
                    initializer=paddle.nn.initializer.Uniform(),
                )
                block.create_parameter(
                    dtype="float32",
                    shape=[5, 10],
                    name="param2",
                    initializer=paddle.nn.initializer.UniformInitializer(
                        seed=456
                    ),
                )
            init_op = block.ops[1]
            self.assertEqual(init_op.attr("seed"), 456)
            init_op1 = block.ops[0]
            self.assertEqual(init_op1.attr("seed"), 123)

    def test_uniform_initializer(self, dtype="float32"):
        """Test uniform initializer with supplied attributes"""
        with paddle.pir_utils.OldIrGuard():
            program = framework.Program()
            block = program.global_block()
            for _ in range(2):
                block.create_parameter(
                    dtype=dtype,
                    shape=[5, 10],
                    name="param",
                    initializer=paddle.nn.initializer.UniformInitializer(
                        -4.2, 3.1, 123
                    ),
                )
            num_ops = 2 if dtype == "float16" else 1
            self.assertEqual(len(block.ops), num_ops)
            init_op = block.ops[0]
            self.assertEqual(init_op.type, 'uniform_random')
            self.assertAlmostEqual(init_op.attr('min'), -4.2, delta=DELTA)
            self.assertAlmostEqual(init_op.attr('max'), 3.1, delta=DELTA)
            self.assertEqual(init_op.attr('seed'), 123)
            return block

    def test_uniform_initializer_two_op(self, dtype="float32"):
        """Test uniform initializer with supplied attributes"""
        with paddle.pir_utils.OldIrGuard():
            program = framework.Program()
            block = program.global_block()
            for i in range(2):
                block.create_parameter(
                    dtype=dtype,
                    shape=[5, 10],
                    name="param",
                    initializer=paddle.nn.initializer.UniformInitializer(
                        -4.2, float(i), 123
                    ),
                )
            num_ops = 2 if dtype == "float16" else 1
            self.assertEqual(len(block.ops), num_ops)
            init_op0 = block.ops[0]
            self.assertEqual(init_op0.type, 'uniform_random')
            self.assertAlmostEqual(init_op0.attr('min'), -4.2, delta=DELTA)
            self.assertAlmostEqual(init_op0.attr('max'), 0.0, delta=DELTA)
            self.assertEqual(init_op0.attr('seed'), 123)
            return block

    def test_uniform_initializer_fp16(self):
        """Test uniform initializer with float16"""
        block = self.test_uniform_initializer_default_value("float16")
        self.assertTrue(check_cast_op(block.ops[1]))
        block = self.test_uniform_initializer(dtype="float16")
        self.assertTrue(check_cast_op(block.ops[1]))
        block = self.test_uniform_initializer_two_op("float16")
        self.assertTrue(check_cast_op(block.ops[1]))

    def test_uniform_initializer_bf16(self):
        """Test uniform initializer with bfloat16
        No cast operator has been added here
        """
        block = self.test_uniform_initializer_default_value("uint16")
        block = self.test_uniform_initializer(dtype="uint16")
        block = self.test_uniform_initializer_two_op("uint16")


class TestUniformInitializerPir(unittest.TestCase):
    def setUp(self):
        self.init_op_name = 'pd_op.uniform'
        self.set_parameter_op_name = 'builtin.set_parameter'

    def get_operand_definition_op_attrs(self, cur_op, operand_name, attr_name):
        input_names = cur_op.get_input_names()
        self.assertIn(operand_name, input_names)
        attr = (
            cur_op.operand(input_names.index(operand_name))
            .source()
            .get_defining_op()
            .attrs()[attr_name]
        )
        return attr

    def test_uniform_initializer_default_value(self, dtype="float32"):
        """Test the uniform initializer with default value"""
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            startup = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                param = paddle.pir.core.create_parameter(
                    dtype=dtype,
                    shape=[5, 10],
                    name="param",
                    initializer=paddle.nn.initializer.Uniform(),
                )
                block = startup.global_block()
                for op in block.ops:
                    # get init op
                    if self.init_op_name == op.name():
                        min = self.get_operand_definition_op_attrs(
                            op, "min", "value"
                        )
                        max = self.get_operand_definition_op_attrs(
                            op, "max", "value"
                        )
                        self.assertAlmostEqual(min, -1.0, delta=DELTA)
                        self.assertAlmostEqual(max, 1.0, delta=DELTA)
                        self.assertEqual(op.attrs()['seed'], 0)

    def test_uniform_initializer_random_seed(self):
        """Test the uniform initializer with manually setting seed"""
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            startup = paddle.static.Program()
            startup.random_seed = 123
            with paddle.static.program_guard(main, startup):
                param1 = paddle.pir.core.create_parameter(
                    dtype="float32",
                    shape=[5, 10],
                    name="param1",
                    initializer=paddle.nn.initializer.Uniform(),
                )

                param2 = paddle.pir.core.create_parameter(
                    dtype="float32",
                    shape=[5, 10],
                    name="param2",
                    initializer=paddle.nn.initializer.UniformInitializer(
                        seed=456
                    ),
                )

                block = startup.global_block()

                checked_parameter_names = []
                for op in block.ops:
                    if self.set_parameter_op_name != op.name():
                        continue

                    parameter_name = op.attrs()["parameter_name"]
                    if parameter_name == "param1":
                        # get "param1"
                        checked_parameter_names.append(parameter_name)
                        seed = (
                            op.operand(0)
                            .source()
                            .get_defining_op()
                            .attrs()['seed']
                        )
                        self.assertEqual(seed, 123)
                    elif parameter_name == "param2":
                        # get "param2"
                        checked_parameter_names.append(parameter_name)
                        seed = (
                            op.operand(0)
                            .source()
                            .get_defining_op()
                            .attrs()['seed']
                        )
                        self.assertEqual(seed, 456)

                self.assertIn("param1", checked_parameter_names)
                self.assertIn("param2", checked_parameter_names)

    def test_uniform_initializer(self, dtype="float32"):
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            startup = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                initializer = paddle.nn.initializer.UniformInitializer(
                    low=-0.5,
                    high=0.5,
                    seed=10,
                    diag_num=16,
                    diag_step=16,
                    diag_val=1.0,
                )
                param = paddle.pir.core.create_parameter(
                    dtype=dtype,
                    shape=[5, 10],
                    name="param",
                    initializer=initializer,
                )
                block = startup.global_block()
                for op in block.ops:
                    # get init op
                    if self.init_op_name == op.name():
                        self.assertEqual(op.attrs()["seed"], 10)

                        input_names = op.get_input_names()
                        self.assertIn('shape', input_names)
                        self.assertIn('min', input_names)
                        self.assertIn('max', input_names)
                        shape = self.get_operand_definition_op_attrs(
                            op, "shape", "value"
                        )
                        min = self.get_operand_definition_op_attrs(
                            op, "min", "value"
                        )
                        max = self.get_operand_definition_op_attrs(
                            op, "max", "value"
                        )
                        self.assertEqual(shape, [5, 10])
                        self.assertAlmostEqual(min, -0.5, DELTA)
                        self.assertAlmostEqual(max, 0.5, DELTA)

    def test_uniform_initializer_fp16(self):
        """Test uniform initializer with float16"""
        self.test_uniform_initializer_default_value(dtype="float16")
        self.test_uniform_initializer(dtype="float16")

    def test_uniform_initializer_bf16(self):
        """Test uniform initializer with float16"""
        self.test_uniform_initializer_default_value(dtype="uint16")
        self.test_uniform_initializer(dtype="uint16")


class TestNormalInitializer(unittest.TestCase):
    def test_normal_initializer_default_value(self):
        """Test the normal initializer with default value"""
        with paddle.pir_utils.OldIrGuard():
            program = framework.Program()
            block = program.global_block()
            for _ in range(2):
                block.create_parameter(
                    dtype="float32",
                    shape=[5, 10],
                    name="param",
                    initializer=paddle.nn.initializer.Normal(),
                )
            self.assertEqual(len(block.ops), 1)
            init_op = block.ops[0]
            self.assertEqual(init_op.type, 'gaussian_random')
            self.assertAlmostEqual(init_op.attr('mean'), 0.0, delta=DELTA)
            self.assertAlmostEqual(init_op.attr('std'), 1.0, delta=DELTA)
            self.assertEqual(init_op.attr('seed'), 0)

    def test_normal_initializer(self, dtype="float32"):
        """Test normal initializer with supplied attributes"""
        with paddle.pir_utils.OldIrGuard():
            program = framework.Program()
            block = program.global_block()
            for _ in range(2):
                block.create_parameter(
                    dtype=dtype,
                    shape=[5, 10],
                    name="param",
                    initializer=paddle.nn.initializer.NormalInitializer(
                        2.3, 1.9, 123
                    ),
                )
            num_ops = 1
            self.assertEqual(len(block.ops), num_ops)
            init_op = block.ops[0]
            self.assertEqual(init_op.type, 'gaussian_random')
            self.assertAlmostEqual(init_op.attr('mean'), 2.3, delta=DELTA)
            self.assertAlmostEqual(init_op.attr('std'), 1.9, delta=DELTA)
            self.assertEqual(init_op.attr('seed'), 123)
            return block

    def test_normal_initializer_complex(self, dtype="complex64"):
        """Test normal initializer with complex dtype"""
        with paddle.pir_utils.OldIrGuard():
            program = framework.Program()
            block = program.global_block()
            for _ in range(2):
                block.create_parameter(
                    dtype=dtype,
                    shape=[5, 10],
                    name="param",
                    initializer=paddle.nn.initializer.NormalInitializer(
                        2.2 + 2.2j, 1.9, 123
                    ),
                )
            num_ops = 1
            self.assertEqual(len(block.ops), num_ops)
            init_op = block.ops[0]
            self.assertEqual(init_op.type, 'gaussian_random')
            self.assertAlmostEqual(init_op.attr('mean'), 2.2, delta=DELTA)
            self.assertAlmostEqual(init_op.attr('std'), 1.9, delta=DELTA)
            self.assertEqual(init_op.attr('seed'), 123)
            return block

    def test_normal_initializer_fp16(self):
        """Test normal initializer with float16"""
        self.test_normal_initializer("float16")

    def test_normal_initializer_bf16(self):
        """Test normal initializer with bfloat16"""
        self.test_normal_initializer("uint16")

    def test_normal_initializer_complex64(self):
        """Test normal initializer with complex64"""
        self.test_normal_initializer_complex("complex64")

    def test_normal_initializer_complex128(self):
        """Test normal initializer with complex128"""
        self.test_normal_initializer_complex("complex128")


class TestXavierInitializer(unittest.TestCase):
    def test_uniform_xavier_initializer(self):
        """Test Xavier initializer with uniform distribution on
        for matrix multiply.
        """
        with paddle.pir_utils.OldIrGuard():
            program = framework.Program()
            block = program.global_block()
            for _ in range(2):
                param = block.create_parameter(
                    dtype="float32",
                    shape=[5, 10],
                    name="param",
                    initializer=paddle.nn.initializer.XavierUniform(),
                )
            self.assertEqual(len(block.ops), 1)
            init_op = block.ops[0]
            self.assertEqual(init_op.type, 'uniform_random')
            limit = np.sqrt(6.0 / (param.shape[0] + param.shape[1]))
            self.assertAlmostEqual(init_op.attr('min'), -limit, delta=DELTA)
            self.assertAlmostEqual(init_op.attr('max'), limit, delta=DELTA)
            self.assertEqual(init_op.attr('seed'), 0)

    def test_uniform_xavier_initializer_conv(self):
        """Test Xavier initializer with uniform distribution on
        for convolutions.
        """
        with paddle.pir_utils.OldIrGuard():
            program = framework.Program()
            block = program.global_block()
            for _ in range(2):
                param = block.create_parameter(
                    dtype="float32",
                    shape=[5, 10, 15, 20],
                    name="param",
                    initializer=paddle.nn.initializer.XavierUniform(),
                )
            self.assertEqual(len(block.ops), 1)
            init_op = block.ops[0]
            self.assertEqual(init_op.type, 'uniform_random')
            receptive_field_size = float(15 * 20)
            limit = np.sqrt(
                6.0 / ((param.shape[0] + param.shape[1]) * receptive_field_size)
            )
            self.assertAlmostEqual(init_op.attr('min'), -limit, delta=DELTA)
            self.assertAlmostEqual(init_op.attr('max'), limit, delta=DELTA)
            self.assertEqual(init_op.attr('seed'), 0)

    def test_normal_xavier_initializer(self):
        """Test Xavier initializer with normal distribution on
        for matrix multiply.
        """
        with paddle.pir_utils.OldIrGuard():
            program = framework.Program()
            block = program.global_block()
            for _ in range(2):
                param = block.create_parameter(
                    dtype="float32",
                    shape=[5, 10],
                    name="param",
                    initializer=paddle.nn.initializer.XavierNormal(),
                )
            self.assertEqual(len(block.ops), 1)
            init_op = block.ops[0]
            self.assertEqual(init_op.type, 'gaussian_random')
            std = np.sqrt(2.0 / (param.shape[0] + param.shape[1]))
            self.assertAlmostEqual(init_op.attr('mean'), 0.0, delta=DELTA)
            self.assertAlmostEqual(init_op.attr('std'), std, delta=DELTA)
            self.assertEqual(init_op.attr('seed'), 0)

    def test_normal_xavier_initializer_conv(self):
        """Test Xavier initializer with normal distribution on
        for convolutions.
        """
        with paddle.pir_utils.OldIrGuard():
            program = framework.Program()
            block = program.global_block()
            for _ in range(2):
                param = block.create_parameter(
                    dtype="float32",
                    shape=[5, 10, 15, 20],
                    name="param",
                    initializer=paddle.nn.initializer.XavierNormal(),
                )
            self.assertEqual(len(block.ops), 1)
            init_op = block.ops[0]
            self.assertEqual(init_op.type, 'gaussian_random')
            receptive_field_size = float(15 * 20)
            std = np.sqrt(
                2.0 / ((param.shape[0] + param.shape[1]) * receptive_field_size)
            )
            self.assertAlmostEqual(init_op.attr('mean'), 0.0, delta=DELTA)
            self.assertAlmostEqual(init_op.attr('std'), std, delta=DELTA)
            self.assertEqual(init_op.attr('seed'), 0)

    def test_xavier_initializer_supplied_arguments(
        self, dtype="float32", uniform=True
    ):
        """Test the Xavier initializer with supplied arguments"""
        with paddle.pir_utils.OldIrGuard():
            program = framework.Program()
            block = program.global_block()
            for _ in range(2):
                block.create_parameter(
                    dtype=dtype,
                    shape=[5, 10],
                    name="param",
                    initializer=paddle.nn.initializer.XavierInitializer(
                        uniform=uniform,
                        fan_in=12,
                        fan_out=23,
                        seed=134,
                        gain=0.2,
                    ),
                )
            num_ops = (
                2
                if (dtype == "float16" or (dtype == "uint16" and not uniform))
                else 1
            )
            self.assertEqual(len(block.ops), num_ops)
            init_op = block.ops[0]
            if uniform:
                self.assertEqual(init_op.type, 'uniform_random')
                limit = 0.2 * np.sqrt(6.0 / (12 + 23))
                self.assertAlmostEqual(init_op.attr('min'), -limit, delta=DELTA)
                self.assertAlmostEqual(init_op.attr('max'), limit, delta=DELTA)
            else:
                self.assertEqual(init_op.type, 'gaussian_random')
            self.assertEqual(init_op.attr('seed'), 134)
            return block

    def test_xavier_initializer_fp16(self):
        """Test the Xavier initializer with float16"""
        block = self.test_xavier_initializer_supplied_arguments("float16")

    def test_xavier_initializer_bf16(self):
        """Test the Xavier initializer with bfloat16"""
        block_uniform = self.test_xavier_initializer_supplied_arguments(
            "uint16"
        )
        self.assertEqual(len(block_uniform.ops), 1)
        block_gaussian = self.test_xavier_initializer_supplied_arguments(
            "uint16", False
        )


class TestXavierInitializerPir(unittest.TestCase):
    def setUp(self):
        self.init_uniform_op_name = 'pd_op.uniform'
        self.init_normal_op_name = 'pd_op.gaussian'
        self.set_parameter_op_name = 'builtin.set_parameter'

    def get_operand_definition_op_attrs(self, cur_op, operand_name, attr_name):
        input_names = cur_op.get_input_names()
        self.assertIn(operand_name, input_names)
        attr = (
            cur_op.operand(input_names.index(operand_name))
            .source()
            .get_defining_op()
            .attrs()[attr_name]
        )
        return attr

    def get_init_ops_by_op_name(self, block, op_name):
        checked_ops = []
        for op in block.ops:
            # get init op
            if op_name == op.name():
                checked_ops.append(op)
        return checked_ops

    def test_uniform_xavier_initializer(self):
        """Test Xavier initializer with uniform distribution on
        for matrix multiply.
        """
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            startup = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                param = paddle.pir.core.create_parameter(
                    dtype="float32",
                    shape=[5, 10],
                    name="param",
                    initializer=paddle.nn.initializer.XavierUniform(),
                )

                block = startup.global_block()
                checked_ops = self.get_init_ops_by_op_name(
                    block, self.init_uniform_op_name
                )
                self.assertEqual(len(checked_ops), 1)
                init_op = checked_ops[0]
                limit = np.sqrt(6.0 / (param.shape[0] + param.shape[1]))
                min = self.get_operand_definition_op_attrs(
                    init_op, "min", "value"
                )
                max = self.get_operand_definition_op_attrs(
                    init_op, "max", "value"
                )
                self.assertAlmostEqual(min, -limit, delta=DELTA)
                self.assertAlmostEqual(max, limit, delta=DELTA)
                self.assertEqual(init_op.attrs()['seed'], 0)

    def test_uniform_xavier_initializer_zero_size(self):
        """Test Xavier initializer with uniform distribution on
        for matrix multiply.
        """
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            startup = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                param = paddle.pir.core.create_parameter(
                    dtype="float32",
                    shape=[0, 0],
                    name="param",
                    initializer=paddle.nn.initializer.XavierUniform(),
                )

                block = startup.global_block()
                checked_ops = self.get_init_ops_by_op_name(
                    block, self.init_uniform_op_name
                )
                self.assertEqual(len(checked_ops), 1)
                init_op = checked_ops[0]
                limit = 0.0
                min = self.get_operand_definition_op_attrs(
                    init_op, "min", "value"
                )
                max = self.get_operand_definition_op_attrs(
                    init_op, "max", "value"
                )
                self.assertAlmostEqual(min, -limit, delta=DELTA)
                self.assertAlmostEqual(max, limit, delta=DELTA)
                self.assertEqual(init_op.attrs()['seed'], 0)

    def test_uniform_xavier_initializer_conv(self):
        """Test Xavier initializer with uniform distribution on
        for convolutions.
        """
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            startup = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                param = paddle.pir.core.create_parameter(
                    dtype="float32",
                    shape=[5, 10, 15, 20],
                    name="param",
                    initializer=paddle.nn.initializer.XavierUniform(),
                )
                block = startup.global_block()
                checked_ops = self.get_init_ops_by_op_name(
                    block, self.init_uniform_op_name
                )
                self.assertEqual(len(checked_ops), 1)
                init_op = checked_ops[0]
                receptive_field_size = float(15 * 20)
                limit = np.sqrt(
                    6.0
                    / ((param.shape[0] + param.shape[1]) * receptive_field_size)
                )
                min = self.get_operand_definition_op_attrs(
                    init_op, "min", "value"
                )
                max = self.get_operand_definition_op_attrs(
                    init_op, "max", "value"
                )
                self.assertAlmostEqual(min, -limit, delta=DELTA)
                self.assertAlmostEqual(max, limit, delta=DELTA)
                self.assertEqual(init_op.attrs()['seed'], 0)

    def test_normal_xavier_initializer(self):
        """Test Xavier initializer with normal distribution on
        for matrix multiply.
        """
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            startup = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                param = paddle.pir.core.create_parameter(
                    dtype="float32",
                    shape=[5, 10],
                    name="param",
                    initializer=paddle.nn.initializer.XavierNormal(),
                )
                block = startup.global_block()
                checked_ops = self.get_init_ops_by_op_name(
                    block, self.init_normal_op_name
                )
                self.assertEqual(len(checked_ops), 1)
                init_op = checked_ops[0]
                std = np.sqrt(2.0 / (param.shape[0] + param.shape[1]))
                self.assertAlmostEqual(
                    init_op.attrs()["mean"], 0.0, delta=DELTA
                )
                self.assertAlmostEqual(init_op.attrs()["std"], std, delta=DELTA)
                self.assertEqual(init_op.attrs()['seed'], 0)

    def test_normal_xavier_initializer_zero_size(self):
        """Test Xavier initializer with normal distribution on
        for matrix multiply.
        """
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            startup = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                param = paddle.pir.core.create_parameter(
                    dtype="float32",
                    shape=[0, 0],
                    name="param",
                    initializer=paddle.nn.initializer.XavierNormal(),
                )
                block = startup.global_block()
                checked_ops = self.get_init_ops_by_op_name(
                    block, self.init_normal_op_name
                )
                self.assertEqual(len(checked_ops), 1)
                init_op = checked_ops[0]
                std = 0.0
                self.assertAlmostEqual(
                    init_op.attrs()["mean"], 0.0, delta=DELTA
                )
                self.assertAlmostEqual(init_op.attrs()["std"], std, delta=DELTA)
                self.assertEqual(init_op.attrs()['seed'], 0)

    def test_normal_xavier_initializer_conv(self):
        """Test Xavier initializer with normal distribution on
        for convolutions.
        """
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            startup = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                param = paddle.pir.core.create_parameter(
                    dtype="float32",
                    shape=[5, 10, 15, 20],
                    name="param",
                    initializer=paddle.nn.initializer.XavierNormal(),
                )
                block = startup.global_block()
                checked_ops = self.get_init_ops_by_op_name(
                    block, self.init_normal_op_name
                )
                self.assertEqual(len(checked_ops), 1)
                init_op = checked_ops[0]
                receptive_field_size = float(15 * 20)
                std = np.sqrt(
                    2.0
                    / ((param.shape[0] + param.shape[1]) * receptive_field_size)
                )
                self.assertAlmostEqual(
                    init_op.attrs()['mean'], 0.0, delta=DELTA
                )
                self.assertAlmostEqual(init_op.attrs()['std'], std, delta=DELTA)
                self.assertEqual(init_op.attrs()['seed'], 0)

    def test_xavier_initializer_supplied_arguments(
        self, dtype="float32", uniform=True
    ):
        """Test the Xavier initializer with supplied arguments"""
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            startup = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                param = paddle.pir.core.create_parameter(
                    dtype=dtype,
                    shape=[5, 10],
                    name="param",
                    initializer=paddle.nn.initializer.XavierInitializer(
                        uniform=uniform,
                        fan_in=12,
                        fan_out=23,
                        seed=134,
                        gain=0.2,
                    ),
                )
                block = startup.global_block()
                init_op_name = (
                    self.init_uniform_op_name
                    if uniform
                    else self.init_normal_op_name
                )

                checked_ops = self.get_init_ops_by_op_name(block, init_op_name)
                self.assertEqual(len(checked_ops), 1)
                init_op = checked_ops[0]
                if uniform:
                    limit = 0.2 * np.sqrt(6.0 / (12 + 23))
                    min = self.get_operand_definition_op_attrs(
                        init_op, "min", "value"
                    )
                    max = self.get_operand_definition_op_attrs(
                        init_op, "max", "value"
                    )
                    self.assertAlmostEqual(min, -limit, delta=DELTA)
                    self.assertAlmostEqual(max, limit, delta=DELTA)

                self.assertEqual(init_op.attrs()['seed'], 134)

        return main, startup

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(), "core is not compiled with CUDA"
    )
    def test_xavier_initializer_fp16(self):
        """Test the Xavier initializer with float16"""
        main_1, startup_1 = self.test_xavier_initializer_supplied_arguments(
            "float16"
        )
        with paddle.pir_utils.IrGuard():
            exe = paddle.static.Executor(paddle.CUDAPlace(0))
            exe.run(startup_1)
            exe.run(main_1)

        main_2, startup_2 = self.test_xavier_initializer_supplied_arguments(
            "float16", uniform=False
        )
        with paddle.pir_utils.IrGuard():
            exe = paddle.static.Executor(paddle.CUDAPlace(0))
            exe.run(startup_2)
            exe.run(main_2)

    @unittest.skipIf(
        not paddle.base.core.is_compiled_with_cuda()
        or not paddle.base.core.is_bfloat16_supported(paddle.CUDAPlace(0)),
        "core is not compiled with CUDA and do not support bfloat16",
    )
    def test_xavier_initializer_bf16(self):
        """Test the Xavier initializer with bfloat16"""
        main_1, startup_1 = self.test_xavier_initializer_supplied_arguments(
            "uint16"
        )
        with paddle.pir_utils.IrGuard():
            exe = paddle.static.Executor(paddle.CUDAPlace(0))
            exe.run(startup_1)
            exe.run(main_1)

        main_2, startup_2 = self.test_xavier_initializer_supplied_arguments(
            "uint16", False
        )
        with paddle.pir_utils.IrGuard():
            exe = paddle.static.Executor(paddle.CUDAPlace(0))
            exe.run(startup_2)
            exe.run(main_2)


class TestMSRAInitializer(unittest.TestCase):
    def test_uniform_msra_initializer(self):
        """Test MSRA initializer with uniform distribution on
        for matrix multiply.
        """
        with paddle.pir_utils.OldIrGuard():
            program = framework.Program()
            block = program.global_block()
            for _ in range(2):
                param = block.create_parameter(
                    dtype="float32",
                    shape=[5, 10],
                    name="param",
                    initializer=paddle.nn.initializer.KaimingUniform(),
                )
            self.assertEqual(len(block.ops), 1)
            init_op = block.ops[0]
            self.assertEqual(init_op.type, 'uniform_random')
            limit = np.sqrt(6.0 / param.shape[0])
            self.assertAlmostEqual(init_op.attr('min'), -limit, delta=DELTA)
            self.assertAlmostEqual(init_op.attr('max'), limit, delta=DELTA)
            self.assertEqual(init_op.attr('seed'), 0)

    def test_uniform_msra_initializer_conv(self):
        """Test MSRA initializer with uniform distribution on
        for convolutions.
        """
        with paddle.pir_utils.OldIrGuard():
            program = framework.Program()
            block = program.global_block()
            for _ in range(2):
                param = block.create_parameter(
                    dtype="float32",
                    shape=[5, 10, 15, 20],
                    name="param",
                    initializer=paddle.nn.initializer.KaimingUniform(),
                )
            self.assertEqual(len(block.ops), 1)
            init_op = block.ops[0]
            self.assertEqual(init_op.type, 'uniform_random')
            receptive_field_size = float(15 * 20)
            limit = np.sqrt(6.0 / (param.shape[1] * receptive_field_size))
            self.assertAlmostEqual(init_op.attr('min'), -limit, delta=DELTA)
            self.assertAlmostEqual(init_op.attr('max'), limit, delta=DELTA)
            self.assertEqual(init_op.attr('seed'), 0)

    def test_normal_msra_initializer(self):
        """Test MSRA initializer with normal distribution on
        for matrix multiply.
        """
        with paddle.pir_utils.OldIrGuard():
            program = framework.Program()
            block = program.global_block()
            for _ in range(2):
                param = block.create_parameter(
                    dtype="float32",
                    shape=[5, 10],
                    name="param",
                    initializer=paddle.nn.initializer.KaimingNormal(),
                )
            self.assertEqual(len(block.ops), 1)
            init_op = block.ops[0]
            self.assertEqual(init_op.type, 'gaussian_random')
            std = np.sqrt(2.0 / param.shape[0])
            self.assertAlmostEqual(init_op.attr('mean'), 0.0, delta=DELTA)
            self.assertAlmostEqual(init_op.attr('std'), std, delta=DELTA)
            self.assertEqual(init_op.attr('seed'), 0)

    def test_normal_msra_initializer_conv(self):
        """Test MSRA initializer with normal distribution on
        for convolutions.
        """
        with paddle.pir_utils.OldIrGuard():
            program = framework.Program()
            block = program.global_block()
            for _ in range(2):
                param = block.create_parameter(
                    dtype="float32",
                    shape=[5, 10, 15, 20],
                    name="param",
                    initializer=paddle.nn.initializer.KaimingNormal(),
                )
            self.assertEqual(len(block.ops), 1)
            init_op = block.ops[0]
            self.assertEqual(init_op.type, 'gaussian_random')
            receptive_field_size = float(15 * 20)
            std = np.sqrt(2.0 / (param.shape[1] * receptive_field_size))
            self.assertAlmostEqual(init_op.attr('mean'), 0.0, delta=DELTA)
            self.assertAlmostEqual(init_op.attr('std'), std, delta=DELTA)
            self.assertEqual(init_op.attr('seed'), 0)

    def test_msra_initializer_supplied_arguments(self, dtype="float32"):
        """Test the MSRA initializer with supplied arguments"""
        with paddle.pir_utils.OldIrGuard():
            program = framework.Program()
            block = program.global_block()
            for _ in range(2):
                block.create_parameter(
                    dtype=dtype,
                    shape=[5, 10],
                    name="param",
                    initializer=paddle.nn.initializer.MSRAInitializer(
                        fan_in=12, seed=134
                    ),
                )
            num_ops = 2 if dtype == "float16" else 1
            self.assertEqual(len(block.ops), num_ops)
            init_op = block.ops[0]
            self.assertEqual(init_op.type, 'uniform_random')
            limit = np.sqrt(6.0 / 12)
            self.assertAlmostEqual(init_op.attr('min'), -limit, delta=DELTA)
            self.assertAlmostEqual(init_op.attr('max'), limit, delta=DELTA)
            self.assertEqual(init_op.attr('seed'), 134)
            return block

    def test_msra_initializer_fp16(self):
        """Test the MSRA initializer with float16"""
        block = self.test_msra_initializer_supplied_arguments("float16")
        self.assertTrue(check_cast_op(block.ops[1]))

    def test_msra_initializer_bf16(self):
        """Test the MSRA initializer with bfloat16"""
        block = self.test_msra_initializer_supplied_arguments("uint16")


class TestMSRAInitializerPir(unittest.TestCase):
    def setUp(self):
        self.init_uniform_op_name = 'pd_op.uniform'
        self.init_normal_op_name = 'pd_op.gaussian'
        self.set_parameter_op_name = 'builtin.set_parameter'

    def get_operand_definition_op_attrs(self, cur_op, operand_name, attr_name):
        input_names = cur_op.get_input_names()
        self.assertIn(operand_name, input_names)
        attr = (
            cur_op.operand(input_names.index(operand_name))
            .source()
            .get_defining_op()
            .attrs()[attr_name]
        )
        return attr

    def get_init_ops_by_op_name(self, block, op_name):
        checked_ops = []
        for op in block.ops:
            # get init op
            if op_name == op.name():
                checked_ops.append(op)
        return checked_ops

    def test_uniform_msra_initializer(self):
        """Test MSRA initializer with uniform distribution on
        for matrix multiply.
        """
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            startup = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                param = paddle.pir.core.create_parameter(
                    dtype="float32",
                    shape=[5, 10],
                    name="param",
                    initializer=paddle.nn.initializer.KaimingUniform(),
                )
                block = startup.global_block()
                checked_ops = self.get_init_ops_by_op_name(
                    block, self.init_uniform_op_name
                )
                self.assertEqual(len(checked_ops), 1)
                init_op = checked_ops[0]
                limit = np.sqrt(6.0 / param.shape[0])
                min = self.get_operand_definition_op_attrs(
                    init_op, "min", "value"
                )
                max = self.get_operand_definition_op_attrs(
                    init_op, "max", "value"
                )
                self.assertAlmostEqual(min, -limit, delta=DELTA)
                self.assertAlmostEqual(max, limit, delta=DELTA)
                self.assertEqual(init_op.attrs()['seed'], 0)

    def test_uniform_msra_initializer_conv(self):
        """Test MSRA initializer with uniform distribution on
        for convolutions.
        """
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            startup = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                param = paddle.pir.core.create_parameter(
                    dtype="float32",
                    shape=[5, 10, 15, 20],
                    name="param",
                    initializer=paddle.nn.initializer.KaimingUniform(),
                )
                block = startup.global_block()
                checked_ops = self.get_init_ops_by_op_name(
                    block, self.init_uniform_op_name
                )
                self.assertEqual(len(checked_ops), 1)
                init_op = checked_ops[0]
                receptive_field_size = float(15 * 20)
                limit = np.sqrt(6.0 / (param.shape[1] * receptive_field_size))
                min = self.get_operand_definition_op_attrs(
                    init_op, "min", "value"
                )
                max = self.get_operand_definition_op_attrs(
                    init_op, "max", "value"
                )
                self.assertAlmostEqual(min, -limit, delta=DELTA)
                self.assertAlmostEqual(max, limit, delta=DELTA)
                self.assertEqual(init_op.attrs()['seed'], 0)

    def test_normal_msra_initializer(self):
        """Test MSRA initializer with normal distribution on
        for matrix multiply.
        """
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            startup = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                param = paddle.pir.core.create_parameter(
                    dtype="float32",
                    shape=[5, 10],
                    name="param",
                    initializer=paddle.nn.initializer.KaimingNormal(),
                )
                block = startup.global_block()
                checked_ops = self.get_init_ops_by_op_name(
                    block, self.init_normal_op_name
                )
                self.assertEqual(len(checked_ops), 1)
                init_op = checked_ops[0]
                std = np.sqrt(2.0 / param.shape[0])
                self.assertAlmostEqual(
                    init_op.attrs()['mean'], 0.0, delta=DELTA
                )
                self.assertAlmostEqual(init_op.attrs()['std'], std, delta=DELTA)
                self.assertEqual(init_op.attrs()['seed'], 0)

    def test_normal_msra_initializer_conv(self):
        """Test MSRA initializer with normal distribution on
        for convolutions.
        """
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            startup = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                param = paddle.pir.core.create_parameter(
                    dtype="float32",
                    shape=[5, 10, 15, 20],
                    name="param",
                    initializer=paddle.nn.initializer.KaimingNormal(),
                )
                block = startup.global_block()
                checked_ops = self.get_init_ops_by_op_name(
                    block, self.init_normal_op_name
                )
                self.assertEqual(len(checked_ops), 1)
                init_op = checked_ops[0]
                receptive_field_size = float(15 * 20)
                std = np.sqrt(2.0 / (param.shape[1] * receptive_field_size))
                self.assertAlmostEqual(
                    init_op.attrs()['mean'], 0.0, delta=DELTA
                )
                self.assertAlmostEqual(init_op.attrs()['std'], std, delta=DELTA)
                self.assertEqual(init_op.attrs()['seed'], 0)

    def test_msra_initializer_supplied_arguments(
        self, dtype="float32", uniform=True
    ):
        """Test the MSRA initializer with supplied arguments"""
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            startup = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                param = paddle.pir.core.create_parameter(
                    dtype=dtype,
                    shape=[5, 10],
                    name="param",
                    initializer=paddle.nn.initializer.MSRAInitializer(
                        fan_in=12, seed=134, uniform=uniform
                    ),
                )
                block = startup.global_block()
                init_op_name = (
                    self.init_uniform_op_name
                    if uniform
                    else self.init_normal_op_name
                )

                checked_ops = self.get_init_ops_by_op_name(block, init_op_name)
                self.assertEqual(len(checked_ops), 1)
                init_op = checked_ops[0]
                if uniform:
                    limit = np.sqrt(6.0 / 12)
                    min = self.get_operand_definition_op_attrs(
                        init_op, "min", "value"
                    )
                    max = self.get_operand_definition_op_attrs(
                        init_op, "max", "value"
                    )
                    self.assertAlmostEqual(min, -limit, delta=DELTA)
                    self.assertAlmostEqual(max, limit, delta=DELTA)

                self.assertEqual(init_op.attrs()['seed'], 134)

        return main, startup

    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(), "core is not compiled with CUDA"
    )
    def test_msra_initializer_fp16(self):
        """Test the MSRA initializer with float16"""
        main_1, startup_1 = self.test_msra_initializer_supplied_arguments(
            "float16"
        )
        with paddle.pir_utils.IrGuard():
            exe = paddle.static.Executor(paddle.CUDAPlace(0))
            exe.run(startup_1)
            exe.run(main_1)

        main_2, startup_2 = self.test_msra_initializer_supplied_arguments(
            "float16", uniform=False
        )
        with paddle.pir_utils.IrGuard():
            exe = paddle.static.Executor(paddle.CUDAPlace(0))
            exe.run(startup_2)
            exe.run(main_2)

    @unittest.skipIf(
        not paddle.base.core.is_compiled_with_cuda()
        or not paddle.base.core.is_bfloat16_supported(paddle.CUDAPlace(0)),
        "core is not compiled with CUDA and do not support bfloat16",
    )
    def test_msra_initializer_bf16(self):
        """Test the MSRA initializer with bfloat16"""
        main_1, startup_1 = self.test_msra_initializer_supplied_arguments(
            "uint16"
        )
        with paddle.pir_utils.IrGuard():
            exe = paddle.static.Executor(paddle.CUDAPlace(0))
            exe.run(startup_1)
            exe.run(main_1)

        main_2, startup_2 = self.test_msra_initializer_supplied_arguments(
            "uint16", uniform=False
        )
        with paddle.pir_utils.IrGuard():
            exe = paddle.static.Executor(paddle.CUDAPlace(0))
            exe.run(startup_2)
            exe.run(main_2)


class TestBilinearInitializer(unittest.TestCase):
    def test_bilinear_initializer(self, dtype="float32"):
        """Test the bilinear initializer with supplied arguments"""
        with paddle.pir_utils.OldIrGuard():
            program = framework.Program()
            block = program.global_block()
            for _ in range(2):
                block.create_parameter(
                    dtype=dtype,
                    shape=[8, 1, 3, 3],
                    name="param",
                    initializer=paddle.nn.initializer.Bilinear(),
                )
            num_ops = 2 if dtype in ["float16", "uint16", "float64"] else 1
            self.assertEqual(len(block.ops), num_ops)
            init_op = block.ops[0]
            self.assertEqual(init_op.type, 'assign_value')
            return block

    def test_bilinear_initializer_fp64(self):
        self.test_bilinear_initializer(dtype='float64')

    def test_bilinear_initializer_fp16(self):
        """Test the bilinear initializer with supplied arguments"""
        block = self.test_bilinear_initializer("float16")
        self.assertTrue(check_cast_op(block.ops[1]))

    def test_bilinear_initializer_bf16(self):
        """Test the bilinear initializer with supplied arguments"""
        block = self.test_bilinear_initializer("uint16")
        self.assertTrue(check_cast_op(block.ops[1]))

    def test_type_error(self):
        self.assertRaises(TypeError, self.test_bilinear_initializer, 'int32')


class TestBilinearInitializerPir(unittest.TestCase):
    def setUp(self):
        self.set_parameter_op_name = 'builtin.set_parameter'
        self.init_op_name = "pd_op.assign_value"
        self.cast_op_name = "pd_op.cast"

    def get_operand_definition_op_attrs(self, cur_op, operand_name, attr_name):
        input_names = cur_op.get_input_names()
        self.assertIn(operand_name, input_names)
        attr = (
            cur_op.operand(input_names.index(operand_name))
            .source()
            .get_defining_op()
            .attrs()[attr_name]
        )
        return attr

    def get_init_ops_by_op_name(self, block, op_name):
        checked_ops = []
        for op in block.ops:
            # get init op
            if op_name == op.name():
                checked_ops.append(op)
        return checked_ops

    def test_bilinear_initializer(self, dtype="float32"):
        """Test the bilinear initializer with supplied arguments"""
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            startup = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                param = paddle.pir.core.create_parameter(
                    dtype=dtype,
                    shape=[8, 1, 3, 3],
                    name="param",
                    initializer=paddle.nn.initializer.Bilinear(),
                )
                block = startup.global_block()
                checked_ops = self.get_init_ops_by_op_name(
                    block, self.init_op_name
                )
                self.assertEqual(len(checked_ops), 1)
                checked_cast_ops = self.get_init_ops_by_op_name(
                    block, self.cast_op_name
                )
                num_cast_op = (
                    1 if dtype in ["float16", "uint16", "float64"] else 0
                )
                self.assertEqual(len(checked_cast_ops), num_cast_op)

            return startup

    def test_bilinear_initializer_fp64(self):
        self.test_bilinear_initializer(dtype='float64')

    def test_bilinear_initializer_fp16(self):
        """Test the bilinear initializer with supplied arguments"""
        startup = self.test_bilinear_initializer("float16")
        cast_ops = self.get_init_ops_by_op_name(
            startup.global_block(), self.cast_op_name
        )
        self.assertGreater(len(cast_ops), 0)
        cast_op = cast_ops[0]
        self.assertTrue(check_cast_op_pir(cast_op))

    def test_bilinear_initializer_bf16(self):
        """Test the bilinear initializer with supplied arguments"""
        startup = self.test_bilinear_initializer("uint16")
        cast_ops = self.get_init_ops_by_op_name(
            startup.global_block(), self.cast_op_name
        )
        self.assertGreater(len(cast_ops), 0)
        cast_op = cast_ops[0]
        self.assertTrue(check_cast_op_pir(cast_op))

    def test_type_error(self):
        self.assertRaises(TypeError, self.test_bilinear_initializer, 'int32')


class TestBilinearInitializerDygraphAPI(unittest.TestCase):
    def func_test_case(self):
        factor = 2
        C = 2
        B = 8
        H = W = 32
        w_attr = paddle.ParamAttr(
            learning_rate=0.0,
            regularizer=L2Decay(0.0),
            initializer=paddle.nn.initializer.Bilinear(),
        )
        data = paddle.rand([B, 3, H, W], dtype='float32')
        conv_up = paddle.nn.Conv2DTranspose(
            3,
            out_channels=C,
            kernel_size=2 * factor - factor % 2,
            padding=int(math.ceil((factor - 1) / 2.0)),
            stride=factor,
            weight_attr=w_attr,
            bias_attr=False,
        )
        x = conv_up(data)
        return x

    def func_test_case_fp16(self):
        paddle.set_default_dtype("float16")
        paddle.seed(1234)
        w_attr = paddle.ParamAttr(
            learning_rate=0.0,
            regularizer=L2Decay(0.0),
            initializer=paddle.nn.initializer.Bilinear(),
        )
        conv2d = paddle.nn.Conv2D(1, 2, 3, weight_attr=w_attr)
        paddle.set_default_dtype("float32")
        return conv2d.weight

    def test_bilinear_initializer(self):
        paddle.disable_static()
        eager_x = self.func_test_case()
        legacy_x = self.func_test_case()
        self.assertEqual(eager_x.numpy().all(), legacy_x.numpy().all())
        paddle.enable_static()

    def test_bilinear_initializer_fp16(self):
        paddle.disable_static()
        eager_x = self.func_test_case_fp16()
        legacy_x = self.func_test_case_fp16()
        self.assertEqual(eager_x.numpy().all(), legacy_x.numpy().all())
        paddle.enable_static()


class TestNumpyArrayInitializer(unittest.TestCase):
    def test_numpy_array_initializer(self, dtype="float32"):
        """Test the numpy array initializer with supplied arguments"""
        import numpy

        with paddle.pir_utils.OldIrGuard():
            program = framework.Program()
            block = program.global_block()
            np_array = numpy.random.random(10000).astype(dtype)
            for _ in range(2):
                block.create_parameter(
                    dtype=np_array.dtype,
                    shape=np_array.shape,
                    name="param",
                    initializer=paddle.nn.initializer.Assign(np_array),
                )
            num_ops = 2 if dtype in ["float16", "uint16"] else 1
            self.assertEqual(len(block.ops), num_ops)
            init_op = block.ops[0]
            self.assertEqual(init_op.type, 'assign_value')
            values = framework.extract_plain_list(init_op.attr('values'))
            assert values == np_array.ravel().tolist()
            return block

    def test_numpy_array_initializer_fp16(self):
        """Test the numpy array initializer with float16"""
        block = self.test_numpy_array_initializer("float16")
        self.assertTrue(block.ops[1])

    def test_numpy_array_initializer_bf16(self):
        """Test the numpy array initializer with bfloat16"""
        block = self.test_numpy_array_initializer("uint16")
        self.assertTrue(block.ops[1])


class TestNumpyArrayInitializerPir(unittest.TestCase):
    def setUp(self):
        self.set_parameter_op_name = 'builtin.set_parameter'
        self.init_op_name = "pd_op.assign_value"
        self.cast_op_name = "pd_op.cast"

    def get_operand_definition_op_attrs(self, cur_op, operand_name, attr_name):
        input_names = cur_op.get_input_names()
        self.assertIn(operand_name, input_names)
        attr = (
            cur_op.operand(input_names.index(operand_name))
            .source()
            .get_defining_op()
            .attrs()[attr_name]
        )
        return attr

    def get_init_ops_by_op_name(self, block, op_name):
        checked_ops = []
        for op in block.ops:
            # get init op
            if op_name == op.name():
                checked_ops.append(op)
        return checked_ops

    def test_numpy_array_initializer(self, dtype="float32"):
        """Test the numpy array initializer with supplied arguments"""
        np_array = np.random.random(10000).astype(dtype)
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            startup = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                param = paddle.pir.core.create_parameter(
                    dtype=np_array.dtype,
                    shape=np_array.shape,
                    name="param",
                    initializer=paddle.nn.initializer.Assign(np_array),
                )
                block = startup.global_block()
                checked_ops = self.get_init_ops_by_op_name(
                    block, self.init_op_name
                )
                self.assertEqual(len(checked_ops), 1)
                checked_cast_ops = self.get_init_ops_by_op_name(
                    block, self.cast_op_name
                )
                num_cast_op = 1 if dtype in ["float16", "uint16"] else 0
                self.assertEqual(len(checked_cast_ops), num_cast_op)

                init_op = checked_ops[0]
                assert (init_op.attrs()['values'] == np_array).all()

            return startup

    def test_numpy_array_initializer_fp16(self):
        """Test the numpy array initializer with float16"""
        startup = self.test_numpy_array_initializer("float16")
        cast_ops = self.get_init_ops_by_op_name(
            startup.global_block(), self.cast_op_name
        )
        self.assertGreater(len(cast_ops), 0)
        cast_op = cast_ops[0]
        self.assertTrue(check_cast_op_pir(cast_op))

    def test_numpy_array_initializer_bf16(self):
        """Test the numpy array initializer with bfloat16"""
        startup = self.test_numpy_array_initializer("uint16")
        cast_ops = self.get_init_ops_by_op_name(
            startup.global_block(), self.cast_op_name
        )
        self.assertGreater(len(cast_ops), 0)
        cast_op = cast_ops[0]
        self.assertTrue(check_cast_op_pir(cast_op))


class TestUniformInitializerDygraph(unittest.TestCase):
    def test_uniform_initializer(self, dtype="float32"):
        """
        In dygraph mode, we can use initializer directly to initialize a tensor.
        """
        paddle.disable_static()

        tensor = paddle.zeros([1024, 1024, 16])
        tensor.stop_gradient = False
        np.testing.assert_allclose(
            np.zeros((1024, 1024, 16)), tensor.numpy(), rtol=1e-05
        )

        uniform_ = paddle.nn.initializer.Uniform()
        uniform_(tensor)

        self.assertEqual(
            tensor.stop_gradient, False
        )  # stop_gradient is not changed

        hist, prob = output_hist(tensor.numpy())

        np.testing.assert_allclose(hist, prob, rtol=0, atol=0.001)

        paddle.enable_static()


class TestXavierInitializerDygraph(unittest.TestCase):
    def test_xavier_initializer(self, dtype="float32"):
        """
        In dygraph mode, we can use initializer directly to initialize a tensor.
        """
        paddle.disable_static()

        tensor = paddle.zeros([1024, 1024, 16])
        tensor.stop_gradient = False

        xavier_ = paddle.nn.initializer.XavierNormal(fan_in=3, fan_out=5)
        xavier_(tensor)

        hist, _ = output_hist(tensor.numpy())

        hist2, _ = output_hist(
            np.random.normal(0, np.sqrt(2.0 / (3 + 5)), [1024, 1024, 16])
        )

        np.testing.assert_allclose(hist, hist2, rtol=0, atol=0.01)
        paddle.enable_static()

    def test_xavier_normal_initializer_zero_size(self, dtype="float32"):
        """
        In dygraph mode, we can use initializer directly to initialize a tensor.
        """
        paddle.disable_static()

        tensor = paddle.zeros([0, 0, 0])
        tensor.stop_gradient = False

        xavier_ = paddle.nn.initializer.XavierNormal(fan_in=0, fan_out=0)
        xavier_(tensor)
        self.assertEqual(tensor.stop_gradient, False)
        self.assertEqual(tensor.shape, [0, 0, 0])

        paddle.enable_static()

    def test_xavier_uniform_initializer_zero_size(self, dtype="float32"):
        """
        In dygraph mode, we can use initializer directly to initialize a tensor.
        """
        paddle.disable_static()

        tensor = paddle.zeros([0, 0, 0])
        tensor.stop_gradient = False

        xavier_ = paddle.nn.initializer.XavierUniform(fan_in=0, fan_out=0)
        xavier_(tensor)
        self.assertEqual(tensor.stop_gradient, False)
        self.assertEqual(tensor.shape, [0, 0, 0])

        paddle.enable_static()


class TestXavierInitializerDygraph2(unittest.TestCase):
    def test_xavier_initializer_with_gain(self, dtype="float32"):
        """
        In dygraph mode, we can use initializer directly to initialize a tensor.
        """
        paddle.disable_static()

        tensor = paddle.zeros([1024, 1024, 16])
        tensor.stop_gradient = False

        xavier_ = paddle.nn.initializer.XavierNormal(
            fan_in=3, fan_out=5, gain=2.5
        )
        xavier_(tensor)

        hist, _ = output_hist(tensor.numpy())

        hist2, _ = output_hist(
            np.random.normal(0, 2.5 * np.sqrt(2.0 / (3 + 5)), [1024, 1024, 16])
        )

        np.testing.assert_allclose(hist, hist2, rtol=0, atol=0.01)
        paddle.enable_static()


class TestMSRAInitializerDygraph(unittest.TestCase):
    def test_msra_initializer(self, dtype="float32"):
        """
        In dygraph mode, we can use initializer directly to initialize a tensor.
        """
        paddle.disable_static()

        tensor = paddle.zeros([1024, 1024, 16])
        tensor.stop_gradient = False

        msra_ = paddle.nn.initializer.KaimingNormal(fan_in=4)
        msra_(tensor)

        hist, _ = output_hist(tensor.numpy())

        hist2, _ = output_hist(
            np.random.normal(0, np.sqrt(2.0 / (4)), [1024, 1024, 16])
        )

        np.testing.assert_allclose(hist, hist2, rtol=0, atol=0.01)
        paddle.enable_static()


class TestConsistencyOfDynamicAndStaticGraph(unittest.TestCase):
    def test_order(self):
        paddle.set_device('cpu')
        SEED = 123
        weight_attr = paddle.framework.ParamAttr(
            name="linear_weight2",
            learning_rate=1.0,
            trainable=False,
            regularizer=None,
            initializer=paddle.nn.initializer.TruncatedNormal(
                mean=0.0, std=2.0
            ),
        )
        bias_attr = paddle.framework.ParamAttr(
            name="linear_bias2",
            learning_rate=1.0,
            trainable=False,
            regularizer=None,
            initializer=paddle.nn.initializer.TruncatedNormal(
                mean=0.0, std=2.0
            ),
        )

        def run_dynamic_graph():
            paddle.seed(SEED)
            linear = paddle.nn.Linear(
                1,
                1,
                weight_attr=paddle.framework.ParamAttr(
                    name="linear_weight1",
                    learning_rate=1.0,
                    trainable=False,
                    regularizer=None,
                    initializer=paddle.nn.initializer.TruncatedNormal(
                        mean=0.0, std=2.0
                    ),
                ),
                bias_attr=paddle.framework.ParamAttr(
                    name="linear_bias1",
                    learning_rate=1.0,
                    trainable=False,
                    regularizer=None,
                    initializer=paddle.nn.initializer.TruncatedNormal(
                        mean=0.0, std=2.0
                    ),
                ),
            )
            return linear.weight.numpy(), linear.bias.numpy()

        def run_static_graph():
            exe = paddle.static.Executor(paddle.CPUPlace())
            paddle.seed(SEED)
            linear = paddle.nn.Linear(
                1, 1, weight_attr=weight_attr, bias_attr=bias_attr
            )
            res = exe.run(
                paddle.static.default_startup_program(),
                fetch_list=[linear.weight, linear.bias],
            )
            return res[0], res[1]

        with dygraph_guard():
            dynamic_res = run_dynamic_graph()
        with static_guard():
            static_res = run_static_graph()

        np.testing.assert_array_equal(dynamic_res[0], static_res[0])
        np.testing.assert_array_equal(dynamic_res[1], static_res[1])

    def test_assign_static_fp32(self):
        random_value = np.random.randn(128, 128).astype("float32")

        def run_dynamic_graph(dtype):
            with dygraph_guard():
                w = paddle.create_parameter(
                    random_value.shape,
                    dtype,
                    default_initializer=paddle.nn.initializer.Assign(
                        random_value
                    ),
                )
            return w

        def run_static_graph(dtype):
            with static_guard():
                exe = paddle.static.Executor(paddle.CPUPlace())
                w = paddle.create_parameter(
                    random_value.shape,
                    dtype,
                    "w",
                    default_initializer=paddle.nn.initializer.Assign(
                        random_value
                    ),
                )
                res = exe.run(
                    paddle.static.default_startup_program(),
                    fetch_list=w,
                )
            return res[0]

        def run_pir_graph(dtype):
            with paddle.pir_utils.IrGuard():
                exe = paddle.static.Executor(paddle.CPUPlace())
                main = paddle.static.Program()
                startup = paddle.static.Program()
                with paddle.static.program_guard(main, startup):
                    param = paddle.pir.core.create_parameter(
                        dtype=dtype,
                        shape=random_value.shape,
                        name="w",
                        initializer=paddle.nn.initializer.Assign(random_value),
                    )
                    exe.run(startup)
                    res = exe.run(main, fetch_list=[param])
            return res[0]

        dynamic_res = run_dynamic_graph("float32")
        static_res = run_static_graph("float32")
        pir_res = run_pir_graph("float32")

        np.testing.assert_array_equal(dynamic_res.numpy(), static_res)
        np.testing.assert_array_equal(dynamic_res.numpy(), pir_res)

    def test_assign_static_fp64(self):
        random_value = np.random.randn(128, 128).astype("float64")

        def run_dynamic_graph(dtype):
            with dygraph_guard():
                w = paddle.create_parameter(
                    random_value.shape,
                    dtype,
                    "www",
                    default_initializer=paddle.nn.initializer.Assign(
                        random_value
                    ),
                )
            return w

        def run_static_graph(dtype):
            with static_guard():
                exe = paddle.static.Executor(paddle.CPUPlace())
                w = paddle.create_parameter(
                    random_value.shape,
                    dtype,
                    "ww",
                    default_initializer=paddle.nn.initializer.Assign(
                        random_value
                    ),
                )
                res = exe.run(
                    paddle.static.default_startup_program(),
                    fetch_list=w,
                )
            return res[0]

        def run_pir_graph(dtype):
            with paddle.pir_utils.IrGuard():
                exe = paddle.static.Executor(paddle.CPUPlace())
                main = paddle.static.Program()
                startup = paddle.static.Program()
                with paddle.static.program_guard(main, startup):
                    param = paddle.pir.core.create_parameter(
                        dtype=dtype,
                        shape=random_value.shape,
                        name="ww",
                        initializer=paddle.nn.initializer.Assign(random_value),
                    )
                    exe.run(startup)
                    res = exe.run(main, fetch_list=[param])
            return res[0]

        dynamic_res = run_dynamic_graph("float64")
        static_res = run_static_graph("float64")
        pir_res = run_pir_graph("float64")

        np.testing.assert_array_equal(dynamic_res.numpy(), static_res)
        np.testing.assert_array_equal(dynamic_res.numpy(), pir_res)


# 2-D Parameter with shape: [10, 15]
class TestOrthogonalInitializer1(unittest.TestCase):
    """
    case 1
    """

    def config(self):
        self.weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Orthogonal(gain=3.0)
        )
        self.dtype = "float64"
        self.in_features = 10
        self.out_features = 15
        self.num_ops = 9

    def check_result(self, a, b):
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(
            np.matmul(a, a.T), 9 * np.eye(10), rtol=1e-5, atol=1e-8
        )

    def test_orthogonal(self):
        self.config()
        paddle.set_default_dtype(self.dtype)

        paddle.disable_static()
        paddle.seed(2021)
        linear = paddle.nn.Linear(
            self.in_features, self.out_features, weight_attr=self.weight_attr
        )
        res_dygraph = linear.weight.numpy()

        paddle.enable_static()
        paddle.seed(2021)
        start_prog = paddle.static.Program()
        main_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, start_prog):
            linear = paddle.nn.Linear(
                self.in_features,
                self.out_features,
                weight_attr=self.weight_attr,
            )

            block = start_prog.global_block()
            if not paddle.framework.use_pir_api():
                self.assertEqual(len(block.ops), self.num_ops)
                self.assertEqual(block.ops[0].type, 'gaussian_random')
                self.assertEqual(block.ops[1].type, 'qr')
                self.assertEqual(block.ops[2].type, 'diag_v2')
                self.assertEqual(block.ops[3].type, 'sign')
                self.assertEqual(block.ops[4].type, 'elementwise_mul')
                self.assertEqual(block.ops[-3].type, 'reshape2')
                self.assertEqual(block.ops[-2].type, 'scale')

            exe = paddle.static.Executor()
            res_static = exe.run(start_prog, fetch_list=[linear.weight])[0]

        self.check_result(res_dygraph, res_static)

    def test_orthogonal_pir(self):
        self.config()
        paddle.set_default_dtype(self.dtype)

        paddle.disable_static()
        paddle.seed(2021)
        linear = paddle.nn.Linear(
            self.in_features, self.out_features, weight_attr=self.weight_attr
        )
        res_dygraph = linear.weight.numpy()

        paddle.enable_static()
        paddle.seed(2021)
        start_prog = paddle.static.Program()
        main_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, start_prog):
            linear = paddle.nn.Linear(
                self.in_features,
                self.out_features,
                weight_attr=self.weight_attr,
            )

            exe = paddle.static.Executor()
            res_static = exe.run(start_prog, fetch_list=[linear.weight])[0]

        self.check_result(res_dygraph, res_static)


# 2-D Parameter with shape: [15, 10]
class TestOrthogonalInitializer2(TestOrthogonalInitializer1):
    """
    case 2
    """

    def config(self):
        self.weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Orthogonal(gain=2.0)
        )
        self.dtype = "float64"
        self.in_features = 15
        self.out_features = 10
        self.num_ops = 8

    def check_result(self, a, b):
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(
            np.matmul(a.T, a), 4 * np.eye(10), rtol=1e-5, atol=1e-8
        )


# 2-D Parameter with shape: [10, 10]
class TestOrthogonalInitializer3(TestOrthogonalInitializer1):
    """
    case 3
    """

    def config(self):
        self.weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Orthogonal()
        )
        self.dtype = "float32"
        self.in_features = 10
        self.out_features = 10
        self.num_ops = 8

    def check_result(self, a, b):
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(
            np.matmul(a.T, a), np.eye(10), rtol=1e-05, atol=1e-06
        )
        np.testing.assert_allclose(
            np.matmul(a, a.T), np.eye(10), rtol=1e-05, atol=1e-06
        )

    def test_error(self):
        self.config()
        with self.assertRaises(AssertionError):
            paddle.nn.Linear(10, 10, bias_attr=self.weight_attr)


# 4-D Parameter with shape: [6, 4, 3, 3]
class TestOrthogonalInitializer4(unittest.TestCase):
    """
    case 4
    """

    def config(self):
        self.weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Orthogonal(gain=3.0)
        )
        self.dtype = "float64"
        self.in_features = 4
        self.out_features = 6
        self.kernel_size = (3, 3)

    def check_result(self, a, b):
        np.testing.assert_array_equal(a, b)
        a = a.reshape(6, -1)
        np.testing.assert_allclose(
            np.matmul(a, a.T), 9 * np.eye(6), rtol=1e-5, atol=1e-8
        )

    def test_orthogonal(self):
        self.config()
        paddle.set_default_dtype(self.dtype)

        paddle.disable_static()
        paddle.seed(2021)
        conv2d = paddle.nn.Conv2D(
            self.in_features,
            self.out_features,
            self.kernel_size,
            weight_attr=self.weight_attr,
        )
        res_dygraph = conv2d.weight.numpy()

        paddle.enable_static()
        paddle.seed(2021)
        start_prog = paddle.static.Program()
        main_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, start_prog):
            inp = paddle.rand(shape=[8, self.in_features, 10, 10])
            conv2d = paddle.nn.Conv2D(
                self.in_features,
                self.out_features,
                self.kernel_size,
                weight_attr=self.weight_attr,
            )
            output = conv2d(inp)
            exe = paddle.static.Executor()

            exe.run(start_prog)
            res_static = exe.run(main_prog, fetch_list=[conv2d.weight])[0]
        self.check_result(res_dygraph, res_static)


# 4-D Parameter with shape: [50, 4, 3, 3]
class TestOrthogonalInitializer5(TestOrthogonalInitializer4):
    """
    case 5
    """

    def config(self):
        self.weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Orthogonal(gain=2.0)
        )
        self.dtype = "float64"
        self.in_features = 4
        self.out_features = 50
        self.kernel_size = (3, 3)

    def check_result(self, a, b):
        np.testing.assert_array_equal(a, b)
        a = a.reshape(50, -1)
        np.testing.assert_allclose(
            np.matmul(a.T, a), 4 * np.eye(36), rtol=1e-5, atol=1e-8
        )


# 4-D Parameter with shape: [36, 4, 3, 3]
class TestOrthogonalInitializer6(TestOrthogonalInitializer4):
    """
    case 6
    """

    def config(self):
        self.weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Orthogonal()
        )
        self.dtype = "float32"
        self.in_features = 4
        self.out_features = 36
        self.kernel_size = (3, 3)

    def check_result(self, a, b):
        np.testing.assert_array_equal(a, b)
        a = a.reshape(36, -1)
        np.testing.assert_allclose(
            np.matmul(a.T, a), np.eye(36), rtol=1e-05, atol=1e-06
        )
        np.testing.assert_allclose(
            np.matmul(a, a.T), np.eye(36), rtol=1e-05, atol=1e-06
        )


# initialize Conv1D weight
class TestDiracInitializer1(unittest.TestCase):
    def config(self):
        self.weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Dirac()
        )
        self.dtype = "float64"
        self.in_channels = 3
        self.out_channels = 2
        self.kernel_size = 3
        self.input_shape = [8, self.in_channels, 10]
        self.conv_layer = paddle.nn.Conv1D
        self.num_ops = (
            8  # fill_constant*2, reshape*2, assign_value*2, scatter, cast
        )

    def check_result(self, w_dygraph, w_static, conv_in, conv_out):
        np.testing.assert_array_equal(w_dygraph, w_static)
        np.testing.assert_array_equal(conv_out, conv_in[:, 0:2, 1:9])

    def test_dirac(self):
        self.config()
        paddle.set_default_dtype(self.dtype)

        paddle.disable_static()
        conv = self.conv_layer(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            weight_attr=self.weight_attr,
        )
        weight_dygraph = conv.weight.numpy()

        paddle.enable_static()
        start_prog = paddle.static.Program()
        main_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, start_prog):
            inp = paddle.rand(self.input_shape)
            conv = self.conv_layer(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                weight_attr=self.weight_attr,
            )

            output = conv(inp)
            block = start_prog.global_block()
            if not paddle.framework.use_pir_api():
                self.assertEqual(len(block.ops), self.num_ops)
                self.assertEqual(block.ops[0].type, 'fill_constant')
                self.assertEqual(block.ops[1].type, 'reshape2')
                self.assertEqual(block.ops[2].type, 'assign_value')
                self.assertEqual(block.ops[3].type, 'assign_value')
                self.assertEqual(block.ops[4].type, 'scatter')
                self.assertEqual(block.ops[5].type, 'reshape2')

            exe = paddle.static.Executor()
            exe.run(start_prog)
            fetch = exe.run(main_prog, fetch_list=[inp, output, conv.weight])
            conv_input = fetch[0]
            conv_output = fetch[1]
            weight_static = fetch[2]

        self.check_result(
            weight_dygraph, weight_static, conv_input, conv_output
        )

    def test_dirac_pir(self):
        self.config()
        paddle.set_default_dtype(self.dtype)

        paddle.disable_static()
        conv = self.conv_layer(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            weight_attr=self.weight_attr,
        )
        weight_dygraph = conv.weight.numpy()

        paddle.enable_static()
        with paddle.pir_utils.IrGuard():
            start_prog = paddle.static.Program()
            main_prog = paddle.static.Program()
            with paddle.static.program_guard(main_prog, start_prog):
                inp = paddle.rand(self.input_shape)
                conv = self.conv_layer(
                    self.in_channels,
                    self.out_channels,
                    self.kernel_size,
                    weight_attr=self.weight_attr,
                )

                output = conv(inp)

                exe = paddle.static.Executor()
                exe.run(start_prog)
                fetch = exe.run(
                    main_prog, fetch_list=[inp, output, conv.weight]
                )
                conv_input = fetch[0]
                conv_output = fetch[1]
                weight_static = fetch[2]

            self.check_result(
                weight_dygraph, weight_static, conv_input, conv_output
            )


# initialize Conv2D weight
class TestDiracInitializer2(TestDiracInitializer1):
    def config(self):
        self.weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Dirac(groups=1)
        )
        self.dtype = "float64"
        self.in_channels = 4
        self.out_channels = 8
        self.kernel_size = (3, 3)
        self.input_shape = [8, self.in_channels, 10, 10]
        self.conv_layer = paddle.nn.Conv2D
        self.num_ops = 8

    def check_result(self, w_dygraph, w_static, conv_in, conv_out):
        np.testing.assert_array_equal(w_dygraph, w_static)
        np.testing.assert_array_equal(
            conv_out[:, 0:4, :, :], conv_in[:, :, 1:9, 1:9]
        )
        np.testing.assert_array_equal(
            conv_out[:, 4:8, :, :], np.zeros([8, 4, 8, 8])
        )


# initialize Conv3D weight
class TestDiracInitializer3(TestDiracInitializer1):
    def config(self):
        self.weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Dirac(groups=2)
        )
        self.dtype = "float32"
        self.in_channels = 5
        self.out_channels = 10
        self.kernel_size = (3, 3, 3)
        self.input_shape = [8, self.in_channels, 10, 10, 10]
        self.conv_layer = paddle.nn.Conv3D
        self.num_ops = 7

    def check_result(self, w_dygraph, w_static, conv_in, conv_out):
        np.testing.assert_array_equal(w_dygraph, w_static)
        np.testing.assert_array_equal(
            conv_out[:, 0:5, :, :, :], conv_in[:, :, 1:9, 1:9, 1:9]
        )
        np.testing.assert_array_equal(
            conv_out[:, 5:10, :, :, :], conv_in[:, :, 1:9, 1:9, 1:9]
        )

    def test_error(self):
        self.config()
        with self.assertRaises(AssertionError):
            paddle.nn.Linear(10, 10, weight_attr=self.weight_attr)

        with self.assertRaises(AssertionError):
            paddle.nn.Conv2D(5, 9, (3, 3), weight_attr=self.weight_attr)


class TestTruncatedNormalInitializerDygraph(unittest.TestCase):
    def _trunc_normal_numpy(self, tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
        # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
        def norm_cdf(x):
            # Computes standard normal cumulative distribution function
            return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        _tensor = np.random.uniform(
            low=2 * l - 1, high=2 * u - 1, size=tensor.shape
        ).astype(paddle.get_default_dtype())

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        _tensor = special.erfinv(_tensor)

        # Transform to proper mean, std
        _tensor = np.multiply(_tensor, std * math.sqrt(2.0))
        _tensor = np.add(_tensor, mean)

        # Clamp to ensure it"s in the proper range
        _tensor = np.clip(_tensor, a_min=a, a_max=b)
        return _tensor

    def test_truncated_normal_initializer_fp32(self):
        """
        In dygraph mode, we can use initializer directly to initialize a tensor.
        """
        with dygraph_guard():
            paddle.seed(42)
            pre_dtype = paddle.get_default_dtype()
            paddle.set_default_dtype("float32")

            tensor = paddle.zeros([1024, 1024, 8])
            tensor.stop_gradient = False

            truncated_normal_ = paddle.nn.initializer.TruncatedNormal()
            truncated_normal_(tensor)

            array = self._trunc_normal_numpy(tensor)
            np.testing.assert_allclose(
                array.mean(), tensor.mean().item(), rtol=0.01, atol=0.01
            )
            np.testing.assert_allclose(
                array.std(), tensor.std().item(), rtol=0.01, atol=0.01
            )
            paddle.set_default_dtype(pre_dtype)

    def test_truncated_normal_initializer_fp64(self):
        """
        In dygraph mode, we can use initializer directly to initialize a tensor.
        """
        with dygraph_guard():
            paddle.seed(42)
            pre_dtype = paddle.get_default_dtype()
            paddle.set_default_dtype("float64")

            tensor = paddle.zeros([1024, 1024, 8])
            tensor.stop_gradient = False

            truncated_normal_ = paddle.nn.initializer.TruncatedNormal()
            truncated_normal_(tensor)

            array = self._trunc_normal_numpy(tensor)
            np.testing.assert_allclose(
                array.mean(), tensor.mean().item(), rtol=0.01, atol=0.01
            )
            np.testing.assert_allclose(
                array.std(), tensor.std().item(), rtol=0.01, atol=0.01
            )
            paddle.set_default_dtype(pre_dtype)


class TestAssignInitializerDygraph(unittest.TestCase):
    def test_assign_initializer_fp32(self):
        """
        In dygraph mode, we can use initializer directly to initialize a tensor.
        """
        with dygraph_guard():
            pre_dtype = paddle.get_default_dtype()
            paddle.set_default_dtype("float32")

            tensor = paddle.zeros(
                [1024, 1024, 8], dtype=paddle.get_default_dtype()
            )
            tensor.stop_gradient = False
            array = np.random.randn(*tensor.shape).astype(
                paddle.get_default_dtype()
            )

            assign_ = paddle.nn.initializer.Assign(array)
            assign_(tensor)

            np.testing.assert_allclose(array, tensor, rtol=1e-6, atol=1e-6)
            paddle.set_default_dtype(pre_dtype)

    def test_assign_initializer_fp64(self):
        """
        In dygraph mode, we can use initializer directly to initialize a tensor.
        """
        with dygraph_guard():
            pre_dtype = paddle.get_default_dtype()
            paddle.set_default_dtype("float64")

            tensor = paddle.zeros(
                [1024, 1024, 8], dtype=paddle.get_default_dtype()
            )
            tensor.stop_gradient = False
            array = np.random.randn(*tensor.shape).astype(
                paddle.get_default_dtype()
            )

            assign_ = paddle.nn.initializer.Assign(array)
            assign_(tensor)

            np.testing.assert_allclose(array, tensor, rtol=1e-6, atol=1e-6)
            paddle.set_default_dtype(pre_dtype)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
