# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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


class TestInverseOp(OpTest):
    def config(self):
        self.matrix_shape = [4, 4]

    def setUp(self):
        self.op_type = "inverse"
        self.config()

        mat = np.random.random(self.matrix_shape).astype("float32")
        inverse = np.linalg.inv(mat)

        self.inputs = {'Input': mat}
        self.outputs = {'Output': inverse}

    def test_check_output(self):
        self.check_output()


class TestInverseOpBatched(TestInverseOp):
    def config(self):
        self.matrix_shape = [4, 4, 4]


if __name__ == "__main__":
    unittest.main()
