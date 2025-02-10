#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest, convert_float_to_uint16

from paddle import enable_static
from paddle.base import core


@unittest.skipIf(
    not core.supports_bfloat16(), "place does not support BF16 evaluation"
)
class TestReshapeBf16Op(OpTest):
    def setUp(self):
        self.op_type = "reshape2"
        self.use_mkldnn = False
        self.mkldnn_data_type = "bfloat16"
        self.init_data()
        self.init_input_data()

        self.inputs = {'X': self.input_data}
        self.attrs = {
            'shape': self.new_shape,
            'use_mkldnn': self.use_mkldnn,
            'mkldnn_data_type': self.mkldnn_data_type,
        }
        self.outputs = {
            "Out": self.inputs["X"].reshape(self.inferred_shape),
            'XShape': np.random.random(self.ori_shape).astype(np.float32),
        }

    def init_data(self):
        self.ori_shape = (10, 2, 6)
        self.new_shape = (10, 0, 3, -1)
        self.inferred_shape = (10, 2, 3, -1)

    def init_input_data(self):
        self.input_data_fp32 = np.random.random(self.ori_shape).astype(
            np.float32
        )
        self.input_data = convert_float_to_uint16(self.input_data_fp32)

    def test_check_output(self):
        self.check_output_with_place(
            core.CPUPlace(),
            no_check_set=['XShape'],
            check_dygraph=False,
            check_pir_onednn=(self.op_type == "reshape2"),
        )

    def test_check_grad(self):
        self.check_grad_with_place(
            core.CPUPlace(),
            ["X"],
            "Out",
            check_dygraph=False,
            user_defined_grads=[self.input_data_fp32],
            user_defined_grad_outputs=[
                self.inputs["X"].reshape(self.inferred_shape)
            ],
            check_pir_onednn=(self.op_type == "reshape2"),
        )


if __name__ == '__main__':
    enable_static()
    unittest.main()
