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

import unittest

import collective.test_communication_api_base as test_base


class TestReshardPToS(test_base.CommunicationTestDistBase):
    def setUp(self):
        super().setUp(num_of_devices=2, timeout=12000)
        self._default_envs = {
            "shape": "(11, 20)",
            "dtype": "float32",
            "seeds": "2024",
        }
        self._changeable_envs = {
            "shard": ["0", "1"],
            "backend": ["gpu"],
            # 新增不同的形状和数据类型来增加测试覆盖率
            # "shape": ["(4, 6)", "(3, 7)"],
            # "dtype": ["float32", "float64"],
        }

    def test_reshard_s_to_p(self):
        envs_list = test_base.gen_product_envs_list(
            self._default_envs, self._changeable_envs
        )
        for envs in envs_list:
            self.run_test_case(
                "pir_reshard_s_to_p.py",
                user_defined_envs=envs,
            )


if __name__ == "__main__":
    unittest.main()
