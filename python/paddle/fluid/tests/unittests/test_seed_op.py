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
from op_test import OpTest
import paddle.fluid as fluid


class TestSeedOpFixSeed(OpTest):
    def setUp(self):
        self.op_type = "seed"
        self.inputs = {}
        self.attrs = {"seed": 123}
        self.outputs = {"Out": np.asarray((123)).astype('int32')}

    def test_check_output(self):
        self.check_output()


class TestSeedOpDiffSeed(OpTest):
    def setUp(self):
        self.op_type = "seed"
        self.inputs = {}
        self.attrs = {"seed": 0}
        self.outputs = {"Out": np.asarray((123)).astype('int32')}

    def test_check_output(self):
        self.check_output(no_check_set=["Out"])


if __name__ == '__main__':
    import paddle
    paddle.enable_static()
    unittest.main()
