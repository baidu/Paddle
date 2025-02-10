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


class TestVPPLocalView(test_base.CommunicationTestDistBase):
    def setUp(self):
        super().setUp(
            num_of_devices=2,
            timeout=300,
        )
        self._default_envs = {"dtype": "float32", "seed": "2024"}
        self._changeable_envs = {"backend": ["gpu"]}

    def test_pp(self):
        self.run_test_case(
            "vpp_pass_unittest_local_view_pir.py",
        )


if __name__ == "__main__":
    unittest.main()
