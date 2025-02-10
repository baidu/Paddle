#!/bin/bash

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

import os
import shlex  # noqa: F401
import sys
import unittest

sys.path.append("../distributed_passes")
from dist_pass_test_base import remove_path_if_exists


class FlPsTest(unittest.TestCase):
    def test_launch_fl_ps(self):
        '''
        cmd = [
            'python', '-m', 'paddle.distributed.fleet.launch', '--log_dir',
            '/ps_log/fl_ps', '--servers', "127.0.0.1:8070", '--workers',
            "127.0.0.1:8080,127.0.0.1:8081", '--heter_workers',
            "127.0.0.1:8090,127.0.0.1:8091", '--heter_devices', "cpu",
            '--worker_num', "2", '--heter_worker_num', "2", 'fl_ps_trainer.py'
        ]
        cmd = [shlex.quote(c) for c in cmd]
        prepare_python_path_and_return_module(__file__)
        exitcode = os.system(' '.join(cmd))
        '''


if __name__ == '__main__':
    remove_path_if_exists('/ps_log')
    remove_path_if_exists('/ps_usr_print_log')
    if not os.path.exists('./train_data'):
        os.system('sh download_data.sh')
        os.system('rm -rf ctr_data.tar.gz')
        os.system('rm -rf train_data_full')
        os.system('rm -rf test_data_full')
    unittest.main()
    if os.path.exists('./train_data'):
        os.system('rm -rf train_data')
        os.system('rm -rf test_data')
