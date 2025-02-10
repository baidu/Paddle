#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()


class XPUTestEinsumOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'einsum'
        self.use_dynamic_create_class = False

    class TestEinsumBinary(XPUOpTest):
        def setUp(self):
            self.op_type = "einsum"
            self.disable = False
            self.types = [self.in_type, self.in_type]
            self.set_mandatory()
            self.init_input()
            np.random.seed(123)
            out = np.einsum(self.equation, *self.inputs)
            self.operands = []
            for idx, inp in enumerate(self.inputs):
                self.operands.append(("x" + str(idx), inp))
            self.inputs = {"Operands": self.operands}
            self.attrs = {"equation": self.equation}
            self.outputs = {
                'Out': out,
                "InnerCache": [
                    ('cache_' + str(i), np.array([1.0]))
                    for i in range(len(self.operands))
                ],
                "XShape": [
                    ('xshape_' + str(i), np.array([1.0]))
                    for i in range(len(self.operands))
                ],
            }

        def init_input(self):
            self.inputs = []
            for t, s in zip(self.types, self.shapes):
                self.inputs.append(np.random.random(s).astype(t))

        def set_mandatory(self):
            self.shapes = [(10, 10, 20), (20, 6)]
            self.equation = "mij,jk->ki"

        def test_check_output(self):
            if not self.disable:
                self.check_output_with_place(
                    paddle.XPUPlace(0),
                    no_check_set=["InnerCache", "XShape"],
                    atol=5,
                    rtol=5
                )

        def test_grad(self):
            if not self.disable:
                self.check_grad_with_place(
                    paddle.XPUPlace(0), [op[0] for op in self.operands], ["Out"]
                )

    class TestEinsum1(TestEinsumBinary):
        def set_mandatory(self):
            self.shapes = [(20, 3, 3), (20, 3, 3)]
            self.equation = "mij,mjk->mik"

    class TestEinsum2(TestEinsumBinary):
        def set_mandatory(self):
            self.shapes = [(20, 3, 3), (20, 3, 3)]
            self.equation = "mij,mjk->ikm"

    class TestEinsum3(TestEinsumBinary):
        def set_mandatory(self):
            self.shapes = [(10, 10), (10, 10)]
            self.equation = "ij,jk->ik"  # }}}

    class TestEinsumWithReduction(TestEinsumBinary):
        def set_mandatory(self):
            self.shapes = [(10, 3, 5), (5, 30)]
            self.equation = "ijk,kl->jl"

    class TestEinsumWithReduction1(TestEinsumBinary):
        def set_mandatory(self):
            self.shapes = [(10, 3, 3, 5), (10, 5, 10, 10)]
            self.equation = "mijk,mklh->ljm"

    class TestEinsumWithUnary(TestEinsumBinary):
        def set_mandatory(self):
            self.shapes = [(10, 10, 3, 5)]
            self.equation = "mijk->mi"

    class TestEinsumWithUnary1(TestEinsumBinary):
        def set_mandatory(self):
            self.shapes = [(5, 10, 3, 3), (3, 6, 3, 10)]
            self.equation = "imjl,jklm->imk"

    class TestEinsumWithBroadcast6(TestEinsumBinary):
        def set_mandatory(self):
            self.shapes = [(100), (100)]
            self.equation = "i,i->"

    class TestEinsumWithDiagonal(TestEinsumBinary):
        def set_mandatory(self):
            self.shapes = [(10, 10)]
            self.equation = "ii->"

    class TestEinsumWithDiagonal2(TestEinsumBinary):
        def set_mandatory(self):
            self.shapes = [(10, 3, 10)]
            self.equation = "iji->j"

    class TestEinsumWithDiagonal5(TestEinsumBinary):
        def set_mandatory(self):
            self.shapes = [(8, 8, 8)]
            self.equation = "aaa->a"

    class TestEinsumWithDiagonal6(TestEinsumBinary):
        def set_mandatory(self):
            self.shapes = [(3, 5, 7, 3), (5, 7, 5, 7)]
            self.equation = "ijki,jkjk->ik"

    class TestEinsumWithDiagonal8(TestEinsumBinary):
        def set_mandatory(self):
            self.shapes = [(3, 5, 7, 3), (5, 7, 5, 7)]
            self.equation = "ijki,jkjk->"


support_types = get_xpu_op_support_types('einsum')
for stype in support_types:
    create_test_class(globals(), XPUTestEinsumOp, stype)


class TestEinsumAPI(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.set_mandatory()

    def test_api(self):
        inputs = []
        for shape, ty in zip(self.shapes, self.types):
            x = paddle.randn(shape).astype(ty)
            x.stop_gradient = False
            inputs.append(x)
        output = paddle.einsum(self.equation, *inputs)
        expect = np.einsum(self.equation, *[x.numpy() for x in inputs])
        np.testing.assert_allclose(
            output.numpy(), expect, atol=1e-2, rtol=5e-2
        )
        output = output.mean()
        output.backward()

    def set_mandatory(self):
        self.shapes = [(10,), (10,)]
        self.types = [np.float32, np.float32]
        self.equation = "...,..."


class TestEinsumWithBroadcast1(TestEinsumAPI):
    def set_mandatory(self):
        self.shapes = [(5, 10, 3, 3)]
        self.types = [np.float32]
        self.equation = "i...->..."


class TestEinsumWithBroadcast2(TestEinsumAPI):
    def set_mandatory(self):
        self.shapes = [(10, 11), (3, 4, 5, 10)]
        self.types = [np.float32, np.float32]
        self.equation = "...ij,...i->j..."


class TestEinsumWithBroadcast4(TestEinsumAPI):
    def set_mandatory(self):
        self.shapes = [(10, 3, 2, 3, 4), (12, 10)]
        self.types = [np.float32, np.float32]
        self.equation = "a...d,...cb->...abcd"


class TestEinsumWithBroadcast5(TestEinsumAPI):
    def set_mandatory(self):
        self.shapes = [(3, 2, 2, 10), (10, 3, 2, 2)]
        self.types = [np.float32, np.float32]
        self.equation = "...a,a...->..."


class TestEinsumWithDiagonal3(TestEinsumAPI):
    def set_mandatory(self):
        self.shapes = [(5, 3, 2, 1, 4, 5)]
        self.types = [np.float32]
        self.equation = "a...a->..."


class TestEinsumWithDiagonal4(TestEinsumAPI):
    def set_mandatory(self):
        self.shapes = [(5, 3, 2, 1, 4, 5)]
        self.types = [np.float32]
        self.equation = "a...a->a..."


if __name__ == "__main__":
    unittest.main()
