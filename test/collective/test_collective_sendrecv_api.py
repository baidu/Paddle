#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from legacy_test.test_collective_api_base import TestDistBase

import paddle

paddle.enable_static()


class TestCollectiveSendRecvAPI(TestDistBase):
    def _setup_config(self):
        pass

    # def test_sendrecv_nccl(self):
    #    if paddle.base.core.is_compiled_with_cuda():
    #        self.check_with_place("collective_sendrecv_api.py", "sendrecv",
    #                              "nccl")

    def test_sendrecv_nccl_with_comm_context(self):
        dtypes_to_test = [
            "float16",
            "float32",
            "float64",
            "int32",
            "int64",
            "int8",
            "uint8",
            "bool",
        ]
        if self._nccl_version >= 21000:
            dtypes_to_test.append("bfloat16")
        for dtype in dtypes_to_test:
            if paddle.base.core.is_compiled_with_cuda():
                self.check_with_place(
                    "collective_sendrecv_api.py",
                    "sendrecv",
                    "nccl",
                    dtype=dtype,
                    need_envs={"USE_COMM_CONTEXT": "1"},
                )

    def test_sendrecv_nccl_with_new_comm_pir(self):
        dtypes_to_test = [
            "float16",
            "float32",
            "float64",
            "int32",
            "int64",
        ]
        if self._nccl_version >= 21000:
            dtypes_to_test.append("bfloat16")
        for dtype in dtypes_to_test:
            if paddle.base.core.is_compiled_with_cuda():
                self.check_with_place(
                    "collective_sendrecv_api.py",
                    "sendrecv",
                    "nccl",
                    dtype=dtype,
                    need_envs={
                        "FLAGS_enable_pir_in_executor": "1",
                    },
                )

    def test_sendrecv_nccl_dygraph(self):
        dtypes_to_test = [
            "float16",
            "float32",
            "float64",
            "int32",
            "int64",
            "int8",
            "uint8",
            "bool",
        ]
        if self._nccl_version >= 21000:
            dtypes_to_test.append("bfloat16")
        for dtype in dtypes_to_test:
            self.check_with_place(
                "collective_sendrecv_api_dygraph.py",
                "sendrecv",
                "nccl",
                static_mode="0",
                dtype=dtype,
            )


if __name__ == "__main__":
    unittest.main()
