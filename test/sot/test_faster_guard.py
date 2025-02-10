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

from __future__ import annotations

import unittest

import paddle


class TestBasicFasterGuard(unittest.TestCase):
    def test_lambda_guard(self):
        guard_lambda = paddle.framework.core.LambdaGuard(lambda x: x == 1)
        self.assertTrue(guard_lambda.check(1))
        self.assertFalse(guard_lambda.check(2))

    def test_type_match_guard(self):
        guard_int = paddle.framework.core.TypeMatchGuard(int)
        guard_str = paddle.framework.core.TypeMatchGuard(str)
        guard_list = paddle.framework.core.TypeMatchGuard(list)
        self.assertTrue(guard_int.check(1))
        self.assertFalse(guard_int.check("1"))
        self.assertTrue(guard_str.check("1"))
        self.assertFalse(guard_str.check(1))
        self.assertTrue(guard_list.check([1]))
        self.assertFalse(guard_list.check(1))

    def test_isinstance_match_guard(self):
        guard_int = paddle.framework.core.InstanceCheckGuard(int)
        guard_str = paddle.framework.core.InstanceCheckGuard(str)
        guard_list = paddle.framework.core.InstanceCheckGuard(list)
        guard_int_bool = paddle.framework.core.InstanceCheckGuard((int, bool))
        guard_int_str = paddle.framework.core.InstanceCheckGuard((int, str))
        self.assertTrue(guard_int.check(1))
        self.assertFalse(guard_int.check("1"))
        self.assertTrue(guard_str.check("1"))
        self.assertFalse(guard_str.check(1))
        self.assertTrue(guard_list.check([1]))
        self.assertFalse(guard_list.check(1))
        self.assertTrue(guard_int_bool.check(1))
        self.assertTrue(guard_int_bool.check(True))
        self.assertFalse(guard_int_bool.check("1"))
        self.assertTrue(guard_int_str.check(1))
        self.assertTrue(guard_int_str.check("1"))
        self.assertTrue(guard_int_str.check(True))
        self.assertFalse(guard_int_str.check([1]))

    def test_value_match_guard(self):
        guard_value = paddle.framework.core.ValueMatchGuard(1)
        guard_container_value = paddle.framework.core.ValueMatchGuard([1])
        self.assertTrue(guard_value.check(1))
        self.assertFalse(guard_value.check(2))
        self.assertTrue(guard_container_value.check([1]))
        self.assertFalse(guard_container_value.check([2]))

    def test_length_match_guard(self):
        guard_length = paddle.framework.core.LengthMatchGuard(1)
        self.assertTrue(guard_length.check([1]))
        self.assertFalse(guard_length.check([1, 2]))

    def test_dtype_match_guard(self):
        guard_dtype = paddle.framework.core.DtypeMatchGuard(paddle.int32)
        self.assertTrue(
            guard_dtype.check(paddle.to_tensor(1, dtype=paddle.int32))
        )
        self.assertFalse(
            guard_dtype.check(paddle.to_tensor(1, dtype=paddle.float32))
        )

    def test_shape_match_guard(self):
        tensor = paddle.randn([2, 3])
        guard_shape = paddle.framework.core.ShapeMatchGuard([2, 3])
        self.assertTrue(guard_shape.check(tensor))
        guard_shape = paddle.framework.core.ShapeMatchGuard([2, None])
        self.assertTrue(guard_shape.check(tensor))
        guard_shape = paddle.framework.core.ShapeMatchGuard([3, 2])
        self.assertFalse(guard_shape.check(tensor))
        guard_shape = paddle.framework.core.ShapeMatchGuard([2, 3, 1])
        self.assertFalse(guard_shape.check(tensor))

    def test_attribute_match_guard(self):
        a = range(1, 10, 2)
        guard_attribute = paddle.framework.core.AttributeMatchGuard(a, "start")
        self.assertTrue(guard_attribute.check(a))
        self.assertFalse(guard_attribute.check(range(10)))

    def test_layer_match_guard(self):
        layer = paddle.nn.Linear(10, 10)
        guard_layer = paddle.framework.core.LayerMatchGuard(layer)
        self.assertTrue(guard_layer.check(layer))
        self.assertFalse(guard_layer.check(paddle.nn.Linear(10, 10)))
        layer.eval()
        self.assertFalse(guard_layer.check(layer))
        layer.train()
        self.assertTrue(guard_layer.check(layer))

    def test_id_match_guard(self):
        layer = paddle.nn.Linear(10, 10)
        guard_id = paddle.framework.core.IdMatchGuard(layer)
        self.assertTrue(guard_id.check(layer))
        layer.eval()
        self.assertTrue(guard_id.check(layer))
        self.assertFalse(guard_id.check(paddle.nn.Linear(10, 10)))


class TestFasterGuardGroup(unittest.TestCase):
    def test_guard_group(self):
        guard_lambda = paddle.framework.core.LambdaGuard(lambda x: x == 1)
        guard_type_match = paddle.framework.core.TypeMatchGuard(int)
        guard_group = paddle.framework.core.GuardGroup(
            [guard_type_match, guard_lambda] * 10
        )
        self.assertTrue(guard_group.check(1))
        self.assertFalse(guard_group.check(2))

    def test_nested_guard_group(self):
        guard_lambda = paddle.framework.core.LambdaGuard(lambda x: x == 1)
        guard_type_match = paddle.framework.core.TypeMatchGuard(int)
        guard_group = paddle.framework.core.GuardGroup(
            [guard_type_match, guard_lambda]
        )
        for _ in range(10):
            guard_group = paddle.framework.core.GuardGroup(
                [guard_group, guard_type_match, guard_lambda]
            )
        self.assertTrue(guard_group.check(1))
        self.assertFalse(guard_group.check(2))

    def test_range_match_guard(self):
        guard_range = paddle.framework.core.RangeMatchGuard(range(1, 10, 2))
        self.assertTrue(guard_range.check(range(1, 10, 2)))
        self.assertFalse(guard_range.check(range(11)))


if __name__ == "__main__":
    unittest.main()
