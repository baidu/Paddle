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

import paddle.fluid as fluid
# from benchmark import BenchmarkSuite
from op_test import OpTest


class TestSumOp(OpTest):
    def setUp(self):
        self.op_type = "sum"
        x0 = np.random.random((3, 4)).astype('float32')
        x1 = np.random.random((3, 4)).astype('float32')
        x2 = np.random.random((3, 4)).astype('float32')

        self.inputs = {"X": [("x0", x0), ("x1", x1), ("x2", x2)]}
        # self.outputs = {"Out": x0 + x1 + x2}
        self.custom_testcase()

    def custom_testcase(self):
        pass

    def test_check_output(self):
        self.check_output(atol=1e-8)

    # def test_check_output_grad(self):
    #     place = fluid.CPUPlace()
    #     self.appends_ops_and_run(place, parallel=False)


if __name__ == "__main__":
    unittest.main()
