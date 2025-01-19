#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import os
import subprocess
import sys
import unittest

sys.path.append(".")


class TestCSoftmaxWithCrossEntropy(unittest.TestCase):
    def pdrun(self, need_envs={}):
        cmd = [
            sys.executable,
            "-m",
            "paddle.distributed.launch",
            "--devices",
            "0,1",
            "c_softmax_with_multi_label_cross_entropy_op.py",
        ]
        envs = os.environ.copy()
        envs.update(need_envs)
        proc = subprocess.Popen(cmd, env=envs)
        return proc

    def test_c_softmax_with_cross_entropy_op(self):
        p = self.pdrun()
        p.wait()


if __name__ == '__main__':
    unittest.main()
