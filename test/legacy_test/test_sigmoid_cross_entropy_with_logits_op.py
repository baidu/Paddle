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


import unittest

import numpy as np
from op_test import OpTest
from scipy.special import expit, logit

import paddle
from paddle import base


def loss_wrapper(
    logit, label, pos_weight=None, normalize=False, ignore_index=-100
):
    out = paddle._C_ops.sigmoid_cross_entropy_with_logits(
        logit, label, pos_weight, normalize, ignore_index
    )
    return out


class TestSigmoidCrossEntropyWithLogitsOp1(OpTest):
    """Test sigmoid_cross_entropy_with_logit_op with binary label"""

    def setUp(self):
        self.op_type = "sigmoid_cross_entropy_with_logits"
        self.python_api = loss_wrapper
        self.prim_op_type = "comp"
        self.public_python_api = loss_wrapper
        batch_size = 64
        num_classes = 20
        self.inputs = {
            'X': logit(
                np.random.uniform(0, 1, (batch_size, num_classes)).astype(
                    "float64"
                )
            ),
            'Label': np.random.randint(0, 2, (batch_size, num_classes)).astype(
                "float64"
            ),
        }

        # Fw Pass is implemented as elementwise sigmoid followed by
        # elementwise logistic loss
        # Label * -log(sigmoid(X)) + (1 - label) * -log(1 - sigmoid(X))
        sigmoid_X = expit(self.inputs['X'])
        term1 = self.inputs['Label'] * np.log(sigmoid_X)
        term2 = (1 - self.inputs['Label']) * np.log(1 - sigmoid_X)
        self.outputs = {'Out': -term1 - term2}

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim_pir=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True, check_prim_pir=True)


class TestSigmoidCrossEntropyWithLogitsOp2(OpTest):
    """Test sigmoid_cross_entropy_with_logit_op with probabilistic label"""

    def setUp(self):
        self.op_type = "sigmoid_cross_entropy_with_logits"
        self.python_api = loss_wrapper
        self.prim_op_type = "comp"
        self.public_python_api = loss_wrapper
        batch_size = 64
        num_classes = 20
        ignore_index = -1
        self.inputs = {
            'X': logit(
                np.random.uniform(0, 1, (batch_size, num_classes)).astype(
                    "float64"
                )
            ),
            'Label': np.random.randint(-1, 2, (batch_size, num_classes)).astype(
                "float64"
            ),
        }
        self.attrs = {
            'ignore_index': ignore_index,
        }
        # Fw Pass is implemented as elementwise sigmoid followed by
        # elementwise logistic loss
        # Label * -log(sigmoid(X)) + (1 - label) * -log(1 - sigmoid(X))
        sigmoid_X = expit(self.inputs['X'])
        term1 = self.inputs['Label'] * np.log(sigmoid_X)
        term2 = (1 - self.inputs['Label']) * np.log(1 - sigmoid_X)
        out = -term1 - term2
        out[np.where(self.inputs['Label'] == ignore_index)] = 0
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim_pir=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True, check_prim_pir=True)


class TestSigmoidCrossEntropyWithLogitsOp3(OpTest):
    """Test sigmoid_cross_entropy_with_logit_op with probabilistic label"""

    def setUp(self):
        self.op_type = "sigmoid_cross_entropy_with_logits"
        self.python_api = loss_wrapper
        self.prim_op_type = "comp"
        self.public_python_api = loss_wrapper
        batch_size = 64
        num_classes = 20
        self.inputs = {
            'X': logit(
                np.random.uniform(0, 1, (batch_size, num_classes)).astype(
                    "float64"
                )
            ),
            'Label': np.random.uniform(0, 1, (batch_size, num_classes)).astype(
                "float64"
            ),
        }

        # Fw Pass is implemented as elementwise sigmoid followed by
        # elementwise logistic loss
        # Label * -log(sigmoid(X)) + (1 - label) * -log(1 - sigmoid(X))
        sigmoid_X = expit(self.inputs['X'])
        term1 = self.inputs['Label'] * np.log(sigmoid_X)
        term2 = (1 - self.inputs['Label']) * np.log(1 - sigmoid_X)
        self.outputs = {'Out': -term1 - term2}

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim_pir=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True, check_prim_pir=True)


class TestSigmoidCrossEntropyWithLogitsOp4(OpTest):
    """Test sigmoid_cross_entropy_with_logit_op with probabilistic label"""

    def setUp(self):
        self.op_type = "sigmoid_cross_entropy_with_logits"
        self.python_api = loss_wrapper
        self.prim_op_type = "comp"
        self.public_python_api = loss_wrapper
        batch_size = 64
        num_classes = 20

        x = logit(
            np.random.uniform(0, 1, (batch_size, num_classes)).astype("float64")
        )
        label = np.random.uniform(0, 1, (batch_size, num_classes)).astype(
            "float64"
        )
        pos_weight = np.random.uniform(0, 1, (batch_size, num_classes)).astype(
            "float64"
        )
        self.inputs = {
            'X': x,
            'Label': label,
            'pos_weight': pos_weight,
        }

        # Fw Pass is implemented as elementwise sigmoid followed by
        # elementwise logistic loss
        max_val = np.clip(-self.inputs['X'], 0, np.finfo(np.float64).max)
        term1 = (1 - label) * self.inputs['X']
        term2 = np.log(np.exp(-max_val) + np.exp(-self.inputs['X'] - max_val))
        out = term1 + pos_weight * (term2 + max_val)

        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim_pir=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', max_relative_error=0.0005, check_pir=True)


class TestSigmoidCrossEntropyWithNorm(OpTest):
    def setUp(self):
        self.op_type = "sigmoid_cross_entropy_with_logits"
        self.python_api = loss_wrapper
        self.prim_op_type = "comp"
        self.public_python_api = loss_wrapper
        batch_size = 64
        num_classes = 20
        ignore_index = -1
        self.inputs = {
            'X': logit(
                np.random.uniform(0, 1, (batch_size, num_classes)).astype(
                    "float64"
                )
            ),
            'Label': np.random.randint(-1, 2, (batch_size, num_classes)).astype(
                "float64"
            ),
        }
        self.attrs = {'ignore_index': ignore_index, 'normalize': True}
        sigmoid_X = expit(self.inputs['X'])
        term1 = self.inputs['Label'] * np.log(sigmoid_X)
        term2 = (1 - self.inputs['Label']) * np.log(1 - sigmoid_X)
        out = -term1 - term2
        out[np.where(self.inputs['Label'] == ignore_index)] = 0
        if self.attrs['normalize']:
            out = out / float(
                np.where(self.inputs['Label'] != ignore_index)[0].size
            )
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim_pir=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True, check_prim_pir=True)


class TestSigmoidCrossEntropyWithLogitsOp5(OpTest):
    """Test sigmoid_cross_entropy_with_logit_op with probabilistic label"""

    def setUp(self):
        self.op_type = "sigmoid_cross_entropy_with_logits"
        self.python_api = loss_wrapper
        self.prim_op_type = "comp"
        self.public_python_api = loss_wrapper
        batch_size = [10, 10]
        num_classes = 20
        self.inputs = {
            'X': logit(
                np.random.uniform(0, 1, (*batch_size, num_classes)).astype(
                    "float64"
                )
            ),
            'Label': np.random.uniform(0, 1, (*batch_size, num_classes)).astype(
                "float64"
            ),
        }

        # Fw Pass is implemented as elementwise sigmoid followed by
        # elementwise logistic loss
        # Label * -log(sigmoid(X)) + (1 - label) * -log(1 - sigmoid(X))
        sigmoid_X = expit(self.inputs['X'])
        term1 = self.inputs['Label'] * np.log(sigmoid_X)
        term2 = (1 - self.inputs['Label']) * np.log(1 - sigmoid_X)
        self.outputs = {'Out': -term1 - term2}

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim_pir=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True, check_prim_pir=True)


class TestSigmoidCrossEntropyWithNorm2(OpTest):
    def setUp(self):
        self.op_type = "sigmoid_cross_entropy_with_logits"
        self.python_api = loss_wrapper
        self.prim_op_type = "comp"
        self.public_python_api = loss_wrapper
        batch_size = [10, 10]
        num_classes = 20
        ignore_index = -1
        self.inputs = {
            'X': logit(
                np.random.uniform(0, 1, (*batch_size, num_classes)).astype(
                    "float64"
                )
            ),
            'Label': np.random.randint(
                -1, 2, (*batch_size, num_classes)
            ).astype("float64"),
        }
        self.attrs = {'ignore_index': ignore_index, 'normalize': True}
        sigmoid_X = expit(self.inputs['X'])
        term1 = self.inputs['Label'] * np.log(sigmoid_X)
        term2 = (1 - self.inputs['Label']) * np.log(1 - sigmoid_X)
        out = -term1 - term2
        out[np.where(self.inputs['Label'] == ignore_index)] = 0
        if self.attrs['normalize']:
            out = out / float(
                np.where(self.inputs['Label'] != ignore_index)[0].size
            )
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim_pir=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True, check_prim_pir=True)


class TestSigmoidCrossEntropyWithLogitsOp6(OpTest):
    """Test sigmoid_cross_entropy_with_logit_op with binary label"""

    def setUp(self):
        self.op_type = "sigmoid_cross_entropy_with_logits"
        self.python_api = loss_wrapper
        self.prim_op_type = "comp"
        self.public_python_api = loss_wrapper
        batch_size = [10, 10]
        num_classes = 20
        self.inputs = {
            'X': logit(
                np.random.uniform(0, 1, (*batch_size, num_classes)).astype(
                    "float64"
                )
            ),
            'Label': np.random.randint(0, 2, (*batch_size, num_classes)).astype(
                "float64"
            ),
        }

        # Fw Pass is implemented as elementwise sigmoid followed by
        # elementwise logistic loss
        # Label * -log(sigmoid(X)) + (1 - label) * -log(1 - sigmoid(X))
        sigmoid_X = expit(self.inputs['X'])
        term1 = self.inputs['Label'] * np.log(sigmoid_X)
        term2 = (1 - self.inputs['Label']) * np.log(1 - sigmoid_X)
        self.outputs = {'Out': -term1 - term2}

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim_pir=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_pir=True, check_prim_pir=True)


class TestSigmoidCrossEntropyWithLogitsOpError(unittest.TestCase):

    def test_errors(self):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):

            def test_Variable():
                # the input of sigmoid_cross_entropy_with_logits must be Variable.
                x1 = base.create_lod_tensor(
                    np.array([-1, 3, 5, 5]),
                    [[1, 1, 1, 1]],
                    base.CPUPlace(),
                )
                lab1 = base.create_lod_tensor(
                    np.array([-1, 3, 5, 5]),
                    [[1, 1, 1, 1]],
                    base.CPUPlace(),
                )
                paddle.nn.functional.binary_cross_entropy_with_logits(x1, lab1)

            self.assertRaises(TypeError, test_Variable)

            def test_dtype():
                # the input dtype of sigmoid_cross_entropy_with_logits must be float16 or float32 or float64
                # float16 only can be set on GPU place
                x2 = paddle.static.data(
                    name='x2', shape=[-1, 3, 4, 5, 6], dtype="int32"
                )
                lab2 = paddle.static.data(
                    name='lab2', shape=[-1, 3, 4, 5, 6], dtype="int32"
                )
                paddle.nn.functional.binary_cross_entropy_with_logits(x2, lab2)

            self.assertRaises(TypeError, test_dtype)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
