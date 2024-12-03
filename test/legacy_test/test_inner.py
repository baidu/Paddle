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


class TestMultiplyApi(unittest.TestCase):
    def _run_static_graph_case(self, x_data, y_data):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            paddle.enable_static()
            x = paddle.static.data(
                name='x', shape=x_data.shape, dtype=x_data.dtype
            )
            y = paddle.static.data(
                name='y', shape=y_data.shape, dtype=y_data.dtype
            )
            res = paddle.inner(x, y)

            place = (
                paddle.CUDAPlace(0)
                if paddle.is_compiled_with_cuda()
                else paddle.CPUPlace()
            )
            exe = paddle.static.Executor(place)
            outs = exe.run(
                paddle.static.default_main_program(),
                feed={'x': x_data, 'y': y_data},
                fetch_list=[res],
            )
            res = outs[0]
            return res

    def _run_dynamic_graph_case(self, x_data, y_data):
        paddle.disable_static()
        x = paddle.to_tensor(x_data)
        y = paddle.to_tensor(y_data)
        res = paddle.inner(x, y)
        return res.numpy()

    def test_multiply_static_case1(self):
        # test static computation graph: 3-d array
        x_data = np.random.rand(2, 10, 10).astype(np.float64)
        y_data = np.random.rand(2, 5, 10).astype(np.float64)
        res = self._run_static_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.inner(x_data, y_data), rtol=1e-05)

    def test_multiply_static_case2(self):
        # test static computation graph: 2-d array
        x_data = np.random.rand(200, 5).astype(np.float64)
        y_data = np.random.rand(50, 5).astype(np.float64)
        res = self._run_static_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.inner(x_data, y_data), rtol=1e-05)

    def test_multiply_static_case3(self):
        # test static computation graph: 1-d array
        x_data = np.random.rand(50).astype(np.float64)
        y_data = np.random.rand(50).astype(np.float64)
        res = self._run_static_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.inner(x_data, y_data), rtol=1e-05)

    def test_multiply_dynamic_case1(self):
        # test dynamic computation graph: 3-d array
        x_data = np.random.rand(5, 10, 10).astype(np.float64)
        y_data = np.random.rand(2, 10).astype(np.float64)
        res = self._run_dynamic_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.inner(x_data, y_data), rtol=1e-05)

    def test_multiply_dynamic_case2(self):
        # test dynamic computation graph: 2-d array
        x_data = np.random.rand(20, 50).astype(np.float64)
        y_data = np.random.rand(50).astype(np.float64)
        res = self._run_dynamic_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.inner(x_data, y_data), rtol=1e-05)

    def test_multiply_dynamic_case3(self):
        # test dynamic computation graph: Scalar
        x_data = np.random.rand(20, 10).astype(np.float32)
        y_data = np.random.rand(1).astype(np.float32).item()
        res = self._run_dynamic_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.inner(x_data, y_data), rtol=1e-05)

    def test_multiply_dynamic_case4(self):
        # test dynamic computation graph: 2-d array Complex
        x_data = np.random.rand(20, 50).astype(
            np.float64
        ) + 1j * np.random.rand(20, 50).astype(np.float64)
        y_data = np.random.rand(50).astype(np.float64) + 1j * np.random.rand(
            50
        ).astype(np.float64)
        res = self._run_dynamic_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.inner(x_data, y_data), rtol=1e-05)

    def test_multiply_dynamic_case5(self):
        # test dynamic computation graph: 3-d array Complex
        x_data = np.random.rand(5, 10, 10).astype(
            np.float64
        ) + 1j * np.random.rand(5, 10, 10).astype(np.float64)
        y_data = np.random.rand(2, 10).astype(np.float64) + 1j * np.random.rand(
            2, 10
        ).astype(np.float64)
        res = self._run_dynamic_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.inner(x_data, y_data), rtol=1e-05)


class TestMultiplyError(unittest.TestCase):

    def test_errors_static_case1(self):
        # test static computation graph: dtype can not be int8
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.static.data(name='x', shape=[100], dtype=np.int8)
            y = paddle.static.data(name='y', shape=[100], dtype=np.int8)
            self.assertRaises(TypeError, paddle.inner, x, y)

    def test_errors_static_case2(self):
        # test static computation graph: inputs must be broadcastable
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.static.data(name='x', shape=[20, 50], dtype=np.float64)
            y = paddle.static.data(name='y', shape=[20], dtype=np.float64)
            self.assertRaises(ValueError, paddle.inner, x, y)

    def test_errors_dynamic_case1(self):
        # test dynamic computation graph: inputs must be broadcastable
        x_data = np.random.rand(20, 5)
        y_data = np.random.rand(10, 2)
        x = paddle.to_tensor(x_data)
        y = paddle.to_tensor(y_data)
        self.assertRaises(Exception, paddle.inner, x, y)

    def test_errors_dynamic_case2(self):
        # test dynamic computation graph: dtype must be Tensor type
        x_data = np.random.randn(200).astype(np.float64)
        y_data = np.random.randn(200).astype(np.float64)
        y = paddle.to_tensor(y_data)
        self.assertRaises(Exception, paddle.inner, x_data, y)

    def test_errors_dynamic_case3(self):
        # test dynamic computation graph: dtype must be Tensor type
        x_data = np.random.randn(200).astype(np.float64)
        y_data = np.random.randn(200).astype(np.float64)
        x = paddle.to_tensor(x_data)
        self.assertRaises(Exception, paddle.inner, x, y_data)

    def test_errors_dynamic_case4(self):
        # test dynamic computation graph: dtype must be Tensor type
        x_data = np.random.randn(200).astype(np.float32)
        y_data = np.random.randn(200).astype(np.float32)
        self.assertRaises(Exception, paddle.inner, x_data, y_data)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
