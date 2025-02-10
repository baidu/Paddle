# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import contextlib
import itertools
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest

import paddle


class CommunicationTestDistBase(unittest.TestCase):
    def setUp(self, save_log_dir=None, num_of_devices=2, timeout=120, nnode=1):
        if num_of_devices > paddle.device.cuda.device_count():
            self.skipTest("number of GPUs is not enough")

        self._python_interp = sys.executable
        self._save_log_dir = save_log_dir
        self._log_dir = tempfile.TemporaryDirectory()
        self._num_of_devices = num_of_devices
        self._device_list = [str(i) for i in range(num_of_devices)]
        self._timeout = timeout
        self._seeds = [i + 10 for i in range(num_of_devices)]
        self._devices = ','.join(self._device_list)
        self._nnode = nnode
        self._port_set = set()

    def _find_free_port(self):
        def __free_port():
            with contextlib.closing(
                socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            ) as s:
                s.bind(('', 0))
                return s.getsockname()[1]

        while True:
            port = __free_port()
            if port not in self._port_set:
                self._port_set.add(port)
                return port

    def run_test_case(self, script_file, user_defined_envs=None):
        runtime_envs = os.environ
        if user_defined_envs is not None:
            runtime_envs.update(user_defined_envs)
        runtime_envs["CUDA_VISIBLE_DEVICES"] = self._devices
        if self._nnode > 1:
            start_command = f"{self._python_interp} -u -m paddle.distributed.launch --nnode={self._nnode} --master=127.0.0.1:{self._find_free_port()} --log_dir {self._log_dir.name} --devices {self._devices} {script_file}"
        else:
            start_command = f"{self._python_interp} -u -m paddle.distributed.launch --log_dir {self._log_dir.name} --devices {self._devices} {script_file}"
        start_command_list = start_command.strip().split()

        if self._nnode > 1:
            for i in range(1, self._nnode):
                p = subprocess.Popen(
                    start_command_list,
                    env=runtime_envs,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

        try:
            self._launcher = subprocess.run(
                start_command_list,
                env=runtime_envs,
                timeout=self._timeout,
                check=True,
            )
        except subprocess.TimeoutExpired as err:
            raise TimeoutError(
                f"Timeout while running command {err.cmd}, try to set a longer period, {err.timeout} is not enough."
            )
        except subprocess.CalledProcessError as err:
            raise RuntimeError(
                f"Error occurs when running this test case. The return code of command {err.cmd} is {err.returncode}"
            )

    def tearDown(self):
        if self._save_log_dir:
            temp_log_dir_name = os.path.basename(self._log_dir.name)
            dir_name = os.path.join(self._save_log_dir, temp_log_dir_name)
            if not os.path.isdir(dir_name):
                print(f"The running logs will copy to {dir_name}")
                shutil.copytree(self._log_dir.name, dir_name)
            else:
                raise RuntimeError(
                    f"Directory {dir_name} exists, failed to save log."
                )


def gen_product_envs_list(default_envs, changeable_envs):
    envs_list = []
    for values in itertools.product(*changeable_envs.values()):
        envs = dict(zip(changeable_envs.keys(), values))
        envs.update(default_envs)
        envs_list.append(envs)
    return envs_list


if __name__ == "__main__":
    unittest.main()
