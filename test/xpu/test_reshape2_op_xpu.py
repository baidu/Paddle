#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from get_test_cover_info import (
    XPUOpTestWrapper,
    check_run_big_shape_test,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()


class XPUTestReshapeOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = "reshape2"
        self.use_dynamic_create_class = False

    # situation 1: have shape( list, no tensor), no actual shape(Tensor)
    class TestReshapeOp(XPUOpTest):
        def setUp(self):
            self.init_data()
            self.op_type = "reshape2"
            self.dtype = self.in_type
            self.init_test_input()
            self.init_test_output()
            self.init_attrs()

        def init_data(self):
            self.ori_shape = (2, 60)
            self.new_shape = (12, 10)
            self.inferred_shape = (12, 10)

        def init_test_input(self):
            self.inputs = {
                "X": np.random.random(self.ori_shape).astype(self.dtype)
            }

        def init_test_output(self):
            self.outputs = {
                "Out": self.inputs["X"].reshape(self.inferred_shape),
                'XShape': np.random.random(self.ori_shape).astype(self.dtype),
            }

        def init_attrs(self):
            self.attrs = {"shape": self.new_shape, "use_xpu": True}

        def test_check_output(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place, no_check_set=['XShape'])

        def test_check_grad(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(place, ["X"], "Out")

    class TestReshapeOpDimInfer1(TestReshapeOp):
        def init_data(self):
            self.ori_shape = (5, 25)
            self.new_shape = (5, -1, 5)
            self.inferred_shape = (5, -1, 5)

    class TestReshapeOpDimInfer2(TestReshapeOp):
        def init_data(self):
            self.ori_shape = (10, 2, 6)
            self.new_shape = (10, 0, 3, -1)
            self.inferred_shape = (10, 2, 3, -1)

    # situation 2: have shape(list, no tensor), have actual shape(Tensor)
    class TestReshapeOpWithInputShape(TestReshapeOp):
        def init_data(self):
            self.ori_shape = (6, 20)
            self.new_shape = (0, -1, 20)
            self.actual_shape = (2, 3, 20)

        def init_test_input(self):
            self.inputs = {
                "X": np.random.random(self.ori_shape).astype(self.dtype),
                "Shape": np.array(self.actual_shape, dtype="int32"),
            }

        def init_test_output(self):
            self.outputs = {
                "Out": self.inputs["X"].reshape(self.actual_shape),
                'XShape': np.random.random(self.ori_shape).astype(self.dtype),
            }

    # Situation 3: have shape(list, have tensor), no actual shape(Tensor)
    class TestReshapeOp_attr_ShapeTensor(TestReshapeOp):
        def init_data(self):
            self.ori_shape = (4, 25)
            self.new_shape = (10, 10)
            self.inferred_shape = (10, 10)
            self.shape = (-1, -1)

        def init_test_input(self):
            shape_tensor = []
            for index, ele in enumerate(self.new_shape):
                shape_tensor.append(
                    ("x" + str(index), np.ones(1).astype('int32') * ele)
                )

            self.inputs = {
                "X": np.random.random(self.ori_shape).astype(self.dtype),
                'ShapeTensor': shape_tensor,
            }

        def init_attrs(self):
            self.attrs = {'shape': self.shape, "use_xpu": True}

    class TestReshapeOpDimInfer1_attr_ShapeTensor(
        TestReshapeOp_attr_ShapeTensor
    ):
        def init_data(self):
            self.ori_shape = (5, 20)
            self.new_shape = (5, -1, 20)
            self.inferred_shape = (5, -1, 20)
            self.shape = (5, -1, -1)

    class TestReshapeOpDimInfer2_attr_ShapeTensor(
        TestReshapeOp_attr_ShapeTensor
    ):
        def init_data(self):
            self.ori_shape = (10, 2, 6)
            self.new_shape = (10, 0, 3, -1)
            self.inferred_shape = (10, 2, 3, -1)
            self.shape = (10, 0, 3, -1)

    # Situation 4: have shape(Tensor), no actual shape(Tensor)
    class TestReshapeOp_attr_OnlyShape(TestReshapeOp):
        def init_data(self):
            self.ori_shape = (4, 25)
            self.new_shape = (10, 10)
            self.inferred_shape = (10, 10)

        def init_test_input(self):
            self.inputs = {
                "X": np.random.random(self.ori_shape).astype(self.dtype),
                "Shape": np.array(self.new_shape, dtype="int32"),
            }

        def init_attrs(self):
            self.attrs = {"use_xpu": True}

    class TestReshapeOpDimInfer1_attr_OnlyShape(TestReshapeOp_attr_OnlyShape):
        def init_data(self):
            self.ori_shape = (5, 20)
            self.new_shape = (5, -1, 10)
            self.inferred_shape = (5, -1, 10)
            self.shape = (5, -1, -1)

    class TestReshapeOpDimInfer2_attr_OnlyShape(TestReshapeOp_attr_OnlyShape):
        def init_data(self):
            self.ori_shape = (10, 2, 6)
            self.new_shape = (10, 0, 3, -1)
            self.inferred_shape = (10, 2, 3, -1)
            self.shape = (10, 0, 3, -1)

    @check_run_big_shape_test()
    class TestReshapeOpLargeShape1(TestReshapeOp):
        def init_data(self):
            self.ori_shape = (5120, 32)
            self.new_shape = (32, 5120)
            self.inferred_shape = (32, 5120)

    @check_run_big_shape_test()
    class TestReshapeOpLargeShape2(TestReshapeOp):
        def init_data(self):
            self.ori_shape = (1, 8192, 5120)
            self.new_shape = (8192, 5120)
            self.inferred_shape = (8192, 5120)

    @check_run_big_shape_test()
    class TestReshapeOpLargeShape3(TestReshapeOp):
        def init_data(self):
            self.ori_shape = (1, 8192)
            self.new_shape = (8192,)
            self.inferred_shape = (8192,)

    @check_run_big_shape_test()
    class TestReshapeOpLargeShape4(TestReshapeOp):
        def init_data(self):
            self.ori_shape = (1, 8192, 5, 64, 2)
            self.new_shape = (1, 8192, 5, 128)
            self.inferred_shape = (1, 8192, 5, 128)

    @check_run_big_shape_test()
    class TestReshapeOpLargeShape5(TestReshapeOp):
        def init_data(self):
            self.ori_shape = (1, 8192, 5, 128)
            self.new_shape = (1, 8192, 640)
            self.inferred_shape = (1, 8192, 640)


support_types = get_xpu_op_support_types("reshape2")
for stype in support_types:
    create_test_class(globals(), XPUTestReshapeOp, stype)

if __name__ == "__main__":
    unittest.main()
