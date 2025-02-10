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

import tempfile
import unittest

import collective.test_communication_api_base as test_base


class TestDPMPPPAPI(test_base.CommunicationTestDistBase):
    def setUp(self):
        super().setUp(num_of_devices=8, timeout=360, nnode=1)
        self._default_envs = {
            "dtype": "float32",
            "seed": "2023",
            "dp": "2",
            "mp": "2",
            "pp": "2",
            "acc_step": "2",
        }
        self._changeable_envs = {
            "backend": ["gpu"],
            "amp": ["true"],
            "amp_level": ["O2"],
            "amp_dtype": ["bfloat16"],
            "amp_master_grad": ["true"],
            "use_lazy_init": ["false"],
            "sequence_parallel": ["false"],
            "prepare_input_output": ["false"],
            "sharding_stage": ["0", "1"],
            "test_share_embedding": [
                "0",
            ],
            "test_position_embedding": [
                "1",
            ],
            "one_api": ["true", "false"],
            "test_lora": ["1"],
        }

    def test_simple_lora_net_dp2_mp2_pp2(self):
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            ckpt_path = tempfile.TemporaryDirectory()
            envs["ckpt_path"] = ckpt_path.name
            self.run_test_case(
                "parallel_api.py",
                user_defined_envs=envs,
            )
            ckpt_path.cleanup()


if __name__ == "__main__":
    unittest.main()
