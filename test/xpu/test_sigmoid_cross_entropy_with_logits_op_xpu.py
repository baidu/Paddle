#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest
from scipy.special import expit, logit

import paddle

paddle.enable_static()


class XPUTestSigmoidCrossEntropyWithLogitsOp(XPUOpTestWrapper):
    """Test sigmoid_cross_entropy_with_logit_op with binary label"""

    def __init__(self):
        self.op_name = "sigmoid_cross_entropy_with_logits"
        self.use_dynamic_create_class = False

    class TestSigmoidCrossEntropyWithLogitsOp(XPUOpTest):
        def setUp(self):
            self.set_xpu()
            self.op_type = "sigmoid_cross_entropy_with_logits"
            self.place = paddle.XPUPlace(0)
            self.init_dtype()
            self.set_inputs()
            self.init_dtype()
            self.set_output()

        def set_output(self):
            # Fw Pass is implemented as elementwise sigmoid followed by
            # elementwise logistic loss
            # Label * -log(sigmoid(X)) + (1 - label) * -log(1 - sigmoid(X))
            sigmoid_X = expit(self.inputs['X'])
            term1 = self.inputs['Label'] * np.log(sigmoid_X)
            term2 = (1 - self.inputs['Label']) * np.log(1 - sigmoid_X)
            self.outputs = {'Out': -term1 - term2}

        def set_inputs(self):
            batch_size = 64
            num_classes = 20
            self.inputs = {
                'X': logit(
                    np.random.uniform(0, 1, (batch_size, num_classes)).astype(
                        self.dtype
                    )
                ),
                'Label': np.random.randint(
                    0, 2, (batch_size, num_classes)
                ).astype(self.dtype),
            }
            self.attrs = {'num_classes': num_classes, 'batch_size': batch_size}

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            self.check_grad_with_place(self.place, ['X'], 'Out')

        def set_xpu(self):
            self.__class__.use_xpu = True
            self.__class__.op_type = self.in_type
            self.place = paddle.XPUPlace(0)

        def init_dtype(self):
            self.dtype = self.in_type

    class TestSigmoidCrossEntropyWithLogitsOp2(
        TestSigmoidCrossEntropyWithLogitsOp
    ):
        """Test sigmoid_cross_entropy_with_logit_op with probabilistic label"""

        def set_inputs(self):
            batch_size = 64
            num_classes = 20
            ignore_index = -1
            self.ignore_index = ignore_index
            self.inputs = {
                'X': logit(
                    np.random.uniform(0, 1, (batch_size, num_classes)).astype(
                        self.dtype
                    )
                ),
                'Label': np.random.randint(
                    -1, 2, (batch_size, num_classes)
                ).astype(self.dtype),
            }
            self.attrs = {'ignore_index': ignore_index}

        def set_output(self):
            # Fw Pass is implemented as elementwise sigmoid followed by
            # elementwise logistic loss
            # Label * -log(sigmoid(X)) + (1 - label) * -log(1 - sigmoid(X))
            sigmoid_X = expit(self.inputs['X'])
            term1 = self.inputs['Label'] * np.log(sigmoid_X)
            term2 = (1 - self.inputs['Label']) * np.log(1 - sigmoid_X)
            out = -term1 - term2
            out[np.where(self.inputs['Label'] == self.ignore_index)] = 0
            self.outputs = {'Out': out}

    class TestSigmoidCrossEntropyWithLogitsOp3(
        TestSigmoidCrossEntropyWithLogitsOp
    ):
        """Test sigmoid_cross_entropy_with_logit_op with probabilistic label"""

        def set_inputs(self):
            batch_size = 64
            num_classes = 20
            self.inputs = {
                'X': logit(
                    np.random.uniform(0, 1, (batch_size, num_classes)).astype(
                        self.dtype
                    )
                ),
                'Label': np.random.uniform(
                    0, 1, (batch_size, num_classes)
                ).astype(self.dtype),
            }
            self.attrs = {'num_classes': num_classes, 'batch_size': batch_size}

        def set_output(self):
            # Fw Pass is implemented as elementwise sigmoid followed by
            # elementwise logistic loss
            # Label * -log(sigmoid(X)) + (1 - label) * -log(1 - sigmoid(X))
            sigmoid_X = expit(self.inputs['X'])
            term1 = self.inputs['Label'] * np.log(sigmoid_X)
            term2 = (1 - self.inputs['Label']) * np.log(1 - sigmoid_X)
            self.outputs = {'Out': -term1 - term2}

    class TestSigmoidCrossEntropyWithLogitsOp4(
        TestSigmoidCrossEntropyWithLogitsOp
    ):
        """Test sigmoid_cross_entropy_with_logit_op with probabilistic label"""

        def set_inputs(self):
            batch_size = 64
            num_classes = 20
            ignore_index = -1
            self.ignore_index = ignore_index
            self.inputs = {
                'X': logit(
                    np.random.uniform(0, 1, (batch_size, num_classes)).astype(
                        self.dtype
                    )
                ),
                'Label': np.random.randint(
                    -1, 2, (batch_size, num_classes)
                ).astype(self.dtype),
            }
            self.attrs = {'ignore_index': ignore_index, 'normalize': True}

        def set_output(self):
            # Fw Pass is implemented as elementwise sigmoid followed by
            # elementwise logistic loss
            # Label * -log(sigmoid(X)) + (1 - label) * -log(1 - sigmoid(X))
            sigmoid_X = expit(self.inputs['X'])
            term1 = self.inputs['Label'] * np.log(sigmoid_X)
            term2 = (1 - self.inputs['Label']) * np.log(1 - sigmoid_X)
            out = -term1 - term2
            out[np.where(self.inputs['Label'] == self.ignore_index)] = 0
            if self.attrs['normalize']:
                out = out / float(
                    np.where(self.inputs['Label'] != self.ignore_index)[0].size
                )
            self.outputs = {'Out': out}

    class TestSigmoidCrossEntropyWithLogitsOp5(
        TestSigmoidCrossEntropyWithLogitsOp
    ):
        """Test sigmoid_cross_entropy_with_logit_op with probabilistic label"""

        def set_inputs(self):
            batch_size = [10, 10]
            num_classes = 20
            self.inputs = {
                'X': logit(
                    np.random.uniform(0, 1, (*batch_size, num_classes)).astype(
                        self.dtype
                    )
                ),
                'Label': np.random.uniform(
                    0, 1, (*batch_size, num_classes)
                ).astype(self.dtype),
            }
            self.attrs = {'num_classes': num_classes, 'batch_size': batch_size}

        def set_output(self):
            # Fw Pass is implemented as elementwise sigmoid followed by
            # elementwise logistic loss
            # Label * -log(sigmoid(X)) + (1 - label) * -log(1 - sigmoid(X))
            sigmoid_X = expit(self.inputs['X'])
            term1 = self.inputs['Label'] * np.log(sigmoid_X)
            term2 = (1 - self.inputs['Label']) * np.log(1 - sigmoid_X)
            self.outputs = {'Out': -term1 - term2}

    class TestSigmoidCrossEntropyWithLogitsOp6(
        TestSigmoidCrossEntropyWithLogitsOp
    ):
        """Test sigmoid_cross_entropy_with_logit_op with binary label"""

        def set_inputs(self):
            batch_size = [10, 10]
            num_classes = 20
            self.inputs = {
                'X': logit(
                    np.random.uniform(0, 1, (*batch_size, num_classes)).astype(
                        self.dtype
                    )
                ),
                'Label': np.random.randint(
                    0, 2, (*batch_size, num_classes)
                ).astype(self.dtype),
            }
            self.attrs = {'num_classes': num_classes, 'batch_size': batch_size}

        def set_output(self):
            # Fw Pass is implemented as elementwise sigmoid followed by
            # elementwise logistic loss
            # Label * -log(sigmoid(X)) + (1 - label) * -log(1 - sigmoid(X))
            sigmoid_X = expit(self.inputs['X'])
            term1 = self.inputs['Label'] * np.log(sigmoid_X)
            term2 = (1 - self.inputs['Label']) * np.log(1 - sigmoid_X)
            self.outputs = {'Out': -term1 - term2}

    class TestSigmoidCrossEntropyWithLogitsOp7(
        TestSigmoidCrossEntropyWithLogitsOp
    ):
        """Test sigmoid_cross_entropy_with_logit_op with binary label"""

        def set_inputs(self):
            batch_size = [10, 10]
            num_classes = 20
            self.inputs = {
                'X': logit(
                    np.random.uniform(0, 1, (*batch_size, num_classes)).astype(
                        self.dtype
                    )
                ),
                'Label': np.random.randint(
                    0, 2, (*batch_size, num_classes)
                ).astype(self.dtype),
                'pos_weight': np.random.uniform(
                    0, 1, (*batch_size, num_classes)
                ).astype(self.dtype),
            }
            self.attrs = {'num_classes': num_classes, 'batch_size': batch_size}

        def set_output(self):
            # Fw Pass is implemented as elementwise sigmoid followed by
            # elementwise logistic loss
            max_val = np.clip(-self.inputs['X'], 0, np.finfo(np.float64).max)
            term1 = (1 - self.inputs['Label']) * self.inputs['X']
            term2 = np.log(
                np.exp(-max_val) + np.exp(-self.inputs['X'] - max_val)
            )
            out = term1 + self.inputs['pos_weight'] * (term2 + max_val)
            self.outputs = {'Out': out}

    class TestSigmoidCrossEntropyWithLogitsNorm(
        TestSigmoidCrossEntropyWithLogitsOp
    ):
        """Test sigmoid_cross_entropy_with_logit_op with probabilistic label"""

        def set_inputs(self):
            batch_size = [10, 10]
            num_classes = 20
            ignore_index = -1
            self.ignore_index = ignore_index
            self.inputs = {
                'X': logit(
                    np.random.uniform(0, 1, (*batch_size, num_classes)).astype(
                        self.dtype
                    )
                ),
                'Label': np.random.randint(
                    -1, 2, (*batch_size, num_classes)
                ).astype(self.dtype),
            }
            self.attrs = {'ignore_index': ignore_index, 'normalize': True}

        def set_output(self):
            # Fw Pass is implemented as elementwise sigmoid followed by
            # elementwise logistic loss
            # Label * -log(sigmoid(X)) + (1 - label) * -log(1 - sigmoid(X))
            sigmoid_X = expit(self.inputs['X'])
            term1 = self.inputs['Label'] * np.log(sigmoid_X)
            term2 = (1 - self.inputs['Label']) * np.log(1 - sigmoid_X)
            out = -term1 - term2
            out[np.where(self.inputs['Label'] == self.ignore_index)] = 0
            if self.attrs['normalize']:
                out = out / float(
                    np.where(self.inputs['Label'] != self.ignore_index)[0].size
                )
            self.outputs = {'Out': out}


support_types = get_xpu_op_support_types('sigmoid_cross_entropy_with_logits')
for stype in support_types:
    create_test_class(globals(), XPUTestSigmoidCrossEntropyWithLogitsOp, stype)

if __name__ == '__main__':
    unittest.main()
