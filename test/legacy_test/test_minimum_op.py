# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle.base import core


class ApiMinimumTest(unittest.TestCase):
    def setUp(self):
        if core.is_compiled_with_cuda():
            self.place = core.CUDAPlace(0)
        else:
            self.place = core.CPUPlace()

        self.input_x = np.random.rand(10, 15).astype("float32")
        self.input_y = np.random.rand(10, 15).astype("float32")
        self.input_z = np.random.rand(15).astype("float32")
        self.input_a = np.array([0, np.nan, np.nan]).astype('int64')
        self.input_b = np.array([2, np.inf, -np.inf]).astype('int64')
        self.input_c = np.array([4, 1, 3]).astype('int64')

        self.np_expected1 = np.minimum(self.input_x, self.input_y)
        self.np_expected2 = np.minimum(self.input_x, self.input_z)
        self.np_expected3 = np.minimum(self.input_a, self.input_c)
        self.np_expected4 = np.minimum(self.input_b, self.input_c)

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            data_x = paddle.static.data("x", shape=[10, 15], dtype="float32")
            data_y = paddle.static.data("y", shape=[10, 15], dtype="float32")
            result_max = paddle.minimum(data_x, data_y)
            exe = paddle.static.Executor(self.place)
            (res,) = exe.run(
                feed={"x": self.input_x, "y": self.input_y},
                fetch_list=[result_max],
            )
        np.testing.assert_allclose(res, self.np_expected1, rtol=1e-05)

        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            data_x = paddle.static.data("x", shape=[10, 15], dtype="float32")
            data_z = paddle.static.data("z", shape=[15], dtype="float32")
            result_max = paddle.minimum(data_x, data_z)
            exe = paddle.static.Executor(self.place)
            (res,) = exe.run(
                feed={"x": self.input_x, "z": self.input_z},
                fetch_list=[result_max],
            )
        np.testing.assert_allclose(res, self.np_expected2, rtol=1e-05)

        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            data_a = paddle.static.data("a", shape=[3], dtype="int64")
            data_c = paddle.static.data("c", shape=[3], dtype="int64")
            result_max = paddle.minimum(data_a, data_c)
            exe = paddle.static.Executor(self.place)
            (res,) = exe.run(
                feed={"a": self.input_a, "c": self.input_c},
                fetch_list=[result_max],
            )
        np.testing.assert_allclose(res, self.np_expected3, rtol=1e-05)

        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            data_b = paddle.static.data("b", shape=[3], dtype="int64")
            data_c = paddle.static.data("c", shape=[3], dtype="int64")
            result_max = paddle.minimum(data_b, data_c)
            exe = paddle.static.Executor(self.place)
            (res,) = exe.run(
                feed={"b": self.input_b, "c": self.input_c},
                fetch_list=[result_max],
            )
        np.testing.assert_allclose(res, self.np_expected4, rtol=1e-05)

    def test_dynamic_api(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.input_x)
        y = paddle.to_tensor(self.input_y)
        z = paddle.to_tensor(self.input_z)

        a = paddle.to_tensor(self.input_a)
        b = paddle.to_tensor(self.input_b)
        c = paddle.to_tensor(self.input_c)

        res = paddle.minimum(x, y)
        res = res.numpy()
        np.testing.assert_allclose(res, self.np_expected1, rtol=1e-05)

        # test broadcast
        res = paddle.minimum(x, z)
        res = res.numpy()
        np.testing.assert_allclose(res, self.np_expected2, rtol=1e-05)

        res = paddle.minimum(a, c)
        res = res.numpy()
        np.testing.assert_allclose(res, self.np_expected3, rtol=1e-05)

        res = paddle.minimum(b, c)
        res = res.numpy()
        np.testing.assert_allclose(res, self.np_expected4, rtol=1e-05)


if __name__ == '__main__':
    unittest.main()
