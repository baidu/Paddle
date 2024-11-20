# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

# FORMAT_VALUE (new)
# BUILD_STRING (new)
from __future__ import annotations

import unittest

from test_case_base import TestCaseBase

import paddle


def set_function(x: paddle.Tensor):
    def inner_function(a, b: int = 3, c: int = 5, *, d: float = 5.0):
        pass

    x = x + 1
    return x


class TestSetFunction(TestCaseBase):
    def test_set_function(self):
        self.assert_results(set_function, paddle.to_tensor(1))


if __name__ == "__main__":
    unittest.main()
