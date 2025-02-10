#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

import paddle
from paddle import base

paddle.enable_static()


class TestCCommInitOp(unittest.TestCase):
    def setUp(self):
        self.endpoints = os.getenv("PADDLE_TRAINER_ENDPOINTS").split(',')
        self.current_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT")
        self.nranks = len(self.endpoints)
        self.rank = self.endpoints.index(self.current_endpoint)
        self.device_id = (
            int(os.getenv("FLAGS_selected_gpus"))
            if not paddle.base.core.is_compiled_with_xpu()
            else int(os.getenv("FLAGS_selected_xpus"))
        )
        self.place = (
            base.CUDAPlace(self.device_id)
            if not paddle.base.core.is_compiled_with_xpu()
            else base.XPUPlace(self.device_id)
        )
        self.exe = base.Executor(self.place)
        self.endpoints.remove(self.current_endpoint)
        self.other_endpoints = self.endpoints

    def test_specifying_devices(self):
        program = base.Program()
        block = program.global_block()
        cl_id_var = block.create_var(
            name=base.unique_name.generate('cl_id'),
            persistable=True,
            type=base.core.VarDesc.VarType.RAW,
        )
        block.append_op(
            type=(
                'c_gen_nccl_id'
                if not paddle.base.core.is_compiled_with_xpu()
                else 'c_gen_bkcl_id'
            ),
            inputs={},
            outputs={'Out': cl_id_var},
            attrs={
                'rank': self.rank,
                'endpoint': self.current_endpoint,
                'other_endpoints': self.other_endpoints,
            },
        )
        block.append_op(
            type='c_comm_init',
            inputs={'X': cl_id_var},
            outputs={},
            attrs={
                'nranks': self.nranks,
                'rank': self.rank,
                'ring_id': 0,
                'device_id': self.device_id,
            },
        )
        self.exe.run(program)


if __name__ == "__main__":
    unittest.main()
