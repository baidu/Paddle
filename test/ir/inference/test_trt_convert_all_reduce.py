# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import paddle


class TestDistTRTWithNewComm(unittest.TestCase):
    def setUp(self):
        self.init_case()
        self.script = "test_trt_all_reduce_infer_script.py"

    def init_case(self):
        self.op_type = "all_reduce"
        self.reduce_type = (0,)
        self.target_value = 4.0
        self.precision = "fp16"

    def test_run(self):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0,1"
        cmd = f"python -u -m paddle.distributed.fleet.launch --gpus 0,1 {self.script} {self.op_type} {self.precision}"
        cmd = cmd.split(" ")

        local_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env)

        local_out, local_err = local_proc.communicate()
        for line in local_out.decode("utf-8").split("\n"):
            results = line.split("=")
            if len(results) == 2 and results[0] == "all_reduce_out":
                self.assertEqual(float(results[1]), self.target_value)

    def test_allreduce_nccl_with_new_comm(self):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0,1"
        env["FLAGS_dynamic_static_unified_comm"] = "1"
        cmd = f"python -u -m paddle.distributed.fleet.launch --gpus 0,1 {self.script} {self.op_type} {self.precision}"
        cmd = cmd.split(" ")

        local_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env)

        local_out, local_err = local_proc.communicate()
        for line in local_out.decode("utf-8").split("\n"):
            results = line.split("=")
            if len(results) == 2 and results[0] == "all_reduce_out":
                self.assertEqual(float(results[1]), self.target_value)


class TestMinWithNewComm(TestDistTRTWithNewComm):
    def init_case(self):
        self.op_type = "all_reduce"
        self.reduce_type = (5,)
        self.target_value = 2.0
        self.precision = "int8"


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
