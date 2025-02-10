# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
import unittest


class TestDistLlamaInference(unittest.TestCase):
    def setUp(self):
        self.script = "dist_llama_inference.py"

    def test_dist_llama_inferece_in_program(self):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0,1"
        env["FLAGS_enable_pir_api"] = "0"
        cmd = f"python -u -m paddle.distributed.launch --gpus 0,1 {self.script}"
        cmd = cmd.split(" ")
        local_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env)

        local_out, local_err = local_proc.communicate()

    def test_dist_llama_inferece_in_pir(self):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0,1"
        env["FLAGS_enable_pir_api"] = "1"
        cmd = f"python -u -m paddle.distributed.launch --gpus 0,1 {self.script}"
        cmd = cmd.split(" ")
        local_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env)

        local_out, local_err = local_proc.communicate()


if __name__ == "__main__":
    unittest.main()  # python run
