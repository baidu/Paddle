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

from __future__ import print_function

import unittest
import numpy as np
import paddle.fluid.core as core
from op_test import OpTest
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard


class TestDropoutWithSeedOp(OpTest):
    def setUp(self):
        self.op_type = "dropout_with_seed"
        self.input_x = np.random.random((32, 64)).astype("float32")
        self.input_seed = np.asarray([0], dtype="int32")
        self.output_out = self.input_x
        self.inputs = {'X': self.input_x, 'Seed': self.input_seed}
        self.attrs = {'dropout_prob': 0.0, 'is_test': False}
        self.outputs = {
            'Out': self.output_out,
            'Mask': np.ones((32, 64)).astype('uint8')
        }

    def test_check_output(self):
        self.check_output(check_dygraph=False)

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out', max_relative_error=0.05)


class TestDropoutWithSeedOp2(TestDropoutWithSeedOp):
    def setUp(self):
        self.op_type = "dropout_with_seed"
        self.input_x = np.random.random((32, 64)).astype("float32")
        self.input_seed = np.asarray([0], dtype="int32")
        self.output_out = np.zeros((32, 64)).astype('float32')
        self.inputs = {'X': self.input_x, 'Seed': self.input_seed}
        self.attrs = {'dropout_prob': 1.0, 'is_test': False}
        self.outputs = {
            'Out': self.output_out,
            'Mask': np.zeros((32, 64)).astype('uint8')
        }


class TestDropoutWithSeedOp3(TestDropoutWithSeedOp):
    def setUp(self):
        self.op_type = "dropout_with_seed"
        self.input_x = np.random.random((32, 64, 2)).astype("float32")
        self.input_seed = np.asarray([0], dtype="int32")
        self.output_out = self.input_x
        self.inputs = {'X': self.input_x, 'Seed': self.input_seed}
        self.attrs = {'dropout_prob': 0.0, 'is_test': False}
        self.outputs = {
            'Out': self.output_out,
            'Mask': np.ones((32, 64, 2)).astype('uint8')
        }


class TestDropoutWithSeedOp4(OpTest):
    def setUp(self):
        self.op_type = "dropout_with_seed"
        self.input_x = np.random.random((32, 64)).astype("float32")
        self.input_seed = np.asarray([0], dtype="int32")
        self.inputs = {'X': self.input_x, 'Seed': self.input_seed}
        self.attrs = {'dropout_prob': 0.35, 'is_test': True}
        self.output_out = self.input_x * (1.0 - self.attrs['dropout_prob'])
        self.outputs = {'Out': self.output_out, }

    def test_check_output(self):
        self.check_output()


class TestDropoutWithSeedOp5(OpTest):
    def setUp(self):
        self.op_type = "dropout_with_seed"
        self.input_x = np.random.random((32, 64, 3)).astype("float32")
        self.input_seed = np.asarray([0], dtype="int32")
        self.inputs = {'X': self.input_x, 'Seed': self.input_seed}
        self.attrs = {'dropout_prob': 0.35, 'is_test': True}
        self.output_out = self.input_x * (1.0 - self.attrs['dropout_prob'])
        self.outputs = {'Out': self.output_out, }

    def test_check_output(self):
        self.check_output()


class TestDropoutWithSeedOp6(TestDropoutWithSeedOp):
    def setUp(self):
        self.op_type = "dropout_with_seed"
        self.input_x = np.random.random((32, 64)).astype("float32")
        self.input_seed = np.asarray([0], dtype="int32")
        self.output_out = np.zeros((32, 64)).astype('float32')
        self.inputs = {'X': self.input_x, 'Seed': self.input_seed}
        self.attrs = {
            'dropout_prob': 1.0,
            'is_test': False,
            'dropout_implementation': 'upscale_in_train'
        }
        self.outputs = {
            'Out': self.output_out,
            'Mask': np.zeros((32, 64)).astype('uint8')
        }


class TestDropoutWithSeedOp7(TestDropoutWithSeedOp):
    def setUp(self):
        self.op_type = "dropout_with_seed"
        self.input_x = np.random.random((32, 64, 2)).astype("float32")
        self.input_seed = np.asarray([0], dtype="int32")
        self.output_out = self.input_x
        self.inputs = {'X': self.input_x, 'Seed': self.input_seed}
        self.attrs = {
            'dropout_prob': 0.0,
            'is_test': False,
            'dropout_implementation': 'upscale_in_train'
        }
        self.outputs = {
            'Out': self.output_out,
            'Mask': np.ones((32, 64, 2)).astype('uint8')
        }


class TestDropoutWithSeedOp8(OpTest):
    def setUp(self):
        self.op_type = "dropout_with_seed"
        self.input_x = np.random.random((32, 64)).astype("float32")
        self.input_seed = np.asarray([0], dtype="int32")
        self.output_out = self.input_x
        self.inputs = {'X': self.input_x, 'Seed': self.input_seed}
        self.attrs = {
            'dropout_prob': 0.35,
            'is_test': True,
            'dropout_implementation': 'upscale_in_train'
        }
        self.outputs = {'Out': self.output_out, }

    def test_check_output(self):
        self.check_output()


class TestDropoutWithSeedOp9(OpTest):
    def setUp(self):
        self.op_type = "dropout_with_seed"
        self.input_x = np.random.random((32, 64, 2)).astype("float32")
        self.input_seed = np.asarray([0], dtype="int32")
        self.output_out = self.input_x
        self.inputs = {'X': self.input_x, 'Seed': self.input_seed}
        self.attrs = {
            'dropout_prob': 0.75,
            'is_test': True,
            'dropout_implementation': 'upscale_in_train'
        }
        self.outputs = {'Out': self.output_out, }

    def test_check_output(self):
        self.check_output()


class TestFP16DropoutWithSeedOp(OpTest):
    def setUp(self):
        self.op_type = "dropout_with_seed"
        self.input_x = np.random.random((32, 64)).astype("float16")
        self.input_seed = np.asarray([0], dtype="int32")
        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(self.input_x),
            'Seed': self.input_seed
        }
        self.attrs = {'dropout_prob': 0.35, 'is_test': True}
        self.output_out = self.input_x * (1.0 - self.attrs['dropout_prob'])
        self.outputs = {'Out': self.output_out, }

    def test_check_output(self):
        if core.is_compiled_with_cuda() and core.op_support_gpu(
                "dropout_with_seed"):
            self.check_output_with_place(core.CUDAPlace(0), atol=1e-3)


class TestFP16DropoutWithSeedOp2(TestFP16DropoutWithSeedOp):
    def setUp(self):
        self.op_type = "dropout_with_seed"
        self.input_x = np.random.random((32, 64, 3)).astype("float16")
        self.input_seed = np.asarray([0], dtype="int32")
        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(self.input_x),
            'Seed': self.input_seed
        }
        self.attrs = {'dropout_prob': 0.75, 'is_test': True}
        self.output_out = self.input_x * (1.0 - self.attrs['dropout_prob'])
        self.outputs = {'Out': self.output_out, }


"""
class TestDropoutOp3(TestDropoutOp):
    def setUp(self):
        self.op_type = "dropout"
        self.inputs = {'X': np.random.random((32, 64, 2)).astype("float32")}
        self.attrs = {'dropout_prob': 0.0, 'fix_seed': True, 'is_test': False}
        self.outputs = {
            'Out': self.inputs['X'],
            'Mask': np.ones((32, 64, 2)).astype('uint8')
        }


class TestDropoutOp4(OpTest):
    def setUp(self):
        self.op_type = "dropout"
        self.inputs = {'X': np.random.random((32, 64)).astype("float32")}
        self.attrs = {'dropout_prob': 0.35, 'fix_seed': True, 'is_test': True}
        self.outputs = {
            'Out': self.inputs['X'] * (1.0 - self.attrs['dropout_prob'])
        }

    def test_check_output(self):
        self.check_output()


class TestDropoutOp5(OpTest):
    def setUp(self):
        self.op_type = "dropout"
        self.inputs = {'X': np.random.random((32, 64, 3)).astype("float32")}
        self.attrs = {'dropout_prob': 0.75, 'is_test': True}
        self.outputs = {
            'Out': self.inputs['X'] * (1.0 - self.attrs['dropout_prob'])
        }

    def test_check_output(self):
        self.check_output()


class TestDropoutOp6(TestDropoutOp):
    def setUp(self):
        self.op_type = "dropout"
        self.inputs = {'X': np.random.random((32, 64)).astype("float32")}
        self.attrs = {
            'dropout_prob': 1.0,
            'fix_seed': True,
            'is_test': False,
            'dropout_implementation': 'upscale_in_train'
        }
        self.outputs = {
            'Out': np.zeros((32, 64)).astype('float32'),
            'Mask': np.zeros((32, 64)).astype('uint8')
        }


class TestDropoutOp7(TestDropoutOp):
    def setUp(self):
        self.op_type = "dropout"
        self.inputs = {'X': np.random.random((32, 64, 2)).astype("float32")}
        self.attrs = {
            'dropout_prob': 0.0,
            'fix_seed': True,
            'is_test': False,
            'dropout_implementation': 'upscale_in_train'
        }
        self.outputs = {
            'Out': self.inputs['X'],
            'Mask': np.ones((32, 64, 2)).astype('uint8')
        }


class TestDropoutOp8(OpTest):
    def setUp(self):
        self.op_type = "dropout"
        self.inputs = {'X': np.random.random((32, 64)).astype("float32")}
        self.attrs = {
            'dropout_prob': 0.35,
            'fix_seed': True,
            'is_test': True,
            'dropout_implementation': 'upscale_in_train'
        }
        self.outputs = {'Out': self.inputs['X']}

    def test_check_output(self):
        self.check_output()


class TestDropoutOp9(OpTest):
    def setUp(self):
        self.op_type = "dropout"
        self.inputs = {'X': np.random.random((32, 64, 3)).astype("float32")}
        self.attrs = {
            'dropout_prob': 0.75,
            'is_test': True,
            'dropout_implementation': 'upscale_in_train'
        }
        self.outputs = {'Out': self.inputs['X']}

    def test_check_output(self):
        self.check_output()


class TestFP16DropoutOp(OpTest):
    def setUp(self):
        self.op_type = "dropout"
        self.init_test_case()

        x = np.random.random(self.input_size).astype("float16")
        out = x * (1.0 - self.prob)
        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.attrs = {
            'dropout_prob': self.prob,
            'fix_seed': self.fix_seed,
            'is_test': True
        }
        self.outputs = {'Out': out}

    def init_test_case(self):
        self.input_size = [32, 64]
        self.prob = 0.35
        self.fix_seed = True

    def test_check_output(self):
        if core.is_compiled_with_cuda() and core.op_support_gpu("dropout"):
            self.check_output_with_place(core.CUDAPlace(0), atol=1e-3)


class TestFP16DropoutOp2(TestFP16DropoutOp):
    def init_test_case(self):
        self.input_size = [32, 64, 3]
        self.prob = 0.75
        self.fix_seed = False


class TestDropoutOpError(OpTest):
    def test_errors(self):
        with program_guard(Program(), Program()):

            def test_Variable():
                # the input of dropout must be Variable.
                x1 = fluid.create_lod_tensor(
                    np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], fluid.CPUPlace())
                fluid.layers.dropout(x1, dropout_prob=0.5)

            self.assertRaises(TypeError, test_Variable)

            def test_dtype():
                # the input dtype of dropout must be float16 or float32 or float64
                # float16 only can be set on GPU place
                x2 = fluid.layers.data(
                    name='x2', shape=[3, 4, 5, 6], dtype="int32")
                fluid.layers.dropout(x2, dropout_prob=0.5)

            self.assertRaises(TypeError, test_dtype)
"""

if __name__ == '__main__':
    unittest.main()
