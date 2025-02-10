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

from __future__ import annotations

import unittest

from test_case_base import TestCaseBase

import paddle
from paddle.jit.sot.psdb import check_no_breakgraph


class ListIterable:
    def __init__(self):
        self._list = [1, 2, 3]

    def __iter__(self):
        return iter(self._list)


@check_no_breakgraph
def list_iterable(x: paddle.Tensor):
    iterable = ListIterable()
    for i in iterable:
        x += i
    return x


class TupleIterable:
    def __init__(self):
        self._tuple = (1, 2, 3)

    def __iter__(self):
        return iter(self._tuple)


@check_no_breakgraph
def tuple_iterable(x: paddle.Tensor):
    iterable = TupleIterable()
    for i in iterable:
        x += i
    return x


class DictIterable:
    def __init__(self):
        self._dict = {0: 1, 1: 2, 2: 3}

    def __iter__(self):
        return iter(self._dict)


@check_no_breakgraph
def dict_iterable(x: paddle.Tensor):
    iterable = DictIterable()
    for i in iterable:
        x += i
    return x


class RangeIterable:
    def __init__(self):
        pass

    def __iter__(self):
        return iter(range(5))


@check_no_breakgraph
def range_iterable(x: paddle.Tensor):
    iterable = RangeIterable()
    for i in iterable:
        x += i
    return x


class TestIterable(TestCaseBase):
    def test_list_iterable(self):
        self.assert_results(list_iterable, paddle.to_tensor(0))

    def test_tuple_iterable(self):
        self.assert_results(tuple_iterable, paddle.to_tensor(0))

    def test_dict_iterable(self):
        self.assert_results(dict_iterable, paddle.to_tensor(0))

    def test_range_iterable(self):
        self.assert_results(range_iterable, paddle.to_tensor(0))


if __name__ == "__main__":
    unittest.main()
