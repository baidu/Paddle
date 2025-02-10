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

import numpy as np

import paddle
from paddle.base import core


class TestSparseIsCoalescedAPI(unittest.TestCase):
    def setUp(self):
        self.dtype = "float32"
        coo_indices = [[0, 0, 0, 1], [0, 0, 1, 2]]
        coo_values = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
        self.coo_tensor = paddle.sparse.sparse_coo_tensor(
            coo_indices, coo_values, dtype=self.dtype
        )
        self.expected_result = False

    def test_is_coalesced(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))

        for place in places:
            paddle.disable_static(place)
            self.assertEqual(
                self.coo_tensor.is_coalesced(), self.expected_result
            )


class TestSparseIsCoalescedAPI1(TestSparseIsCoalescedAPI):
    def setUp(self):
        self.dtype = "float64"
        coo_indices = [[0, 0, 1, 2], [0, 1, 1, 2]]
        coo_values = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
        self.coo_tensor = paddle.sparse.sparse_coo_tensor(
            coo_indices, coo_values, dtype=self.dtype
        )
        self.expected_result = False


class TestSparseIsCoalescedAPI2(TestSparseIsCoalescedAPI):
    def setUp(self):
        coo_indices = [[0, 0, 1, 2], [0, 2, 0, 2], [0, 1, 1, 0]]
        coo_values = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
        self.dtype = "int16"
        self.coo_tensor = paddle.sparse.sparse_coo_tensor(
            coo_indices, coo_values, dtype=self.dtype
        ).coalesce()
        self.expected_result = True


class TestSparseIsCoalescedAPI3(TestSparseIsCoalescedAPI):
    def setUp(self):
        coo_indices = [[0, 0, 0, 1], [0, 0, 1, 2]]
        coo_values = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
        self.dtype = "int32"
        self.coo_tensor = paddle.sparse.sparse_coo_tensor(
            coo_indices, coo_values, dtype=self.dtype
        ).coalesce()
        self.expected_result = True


class TestSparseIsCoalescedAPI4(TestSparseIsCoalescedAPI):
    def setUp(self):
        coo_indices = [[0, 0, 0, 1], [0, 0, 1, 2]]
        coo_values = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
        self.dtype = "int64"
        self.coo_tensor = paddle.sparse.sparse_coo_tensor(
            coo_indices, coo_values, dtype=self.dtype
        )
        self.expected_result = False


class TestSparseIsCoalescedAPI5(TestSparseIsCoalescedAPI):
    def setUp(self):
        coo_indices = [[0, 0, 0, 1], [0, 0, 1, 2]]
        coo_values = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
        self.dtype = "uint8"
        self.coo_tensor = paddle.sparse.sparse_coo_tensor(
            coo_indices, coo_values, dtype=self.dtype
        )
        self.expected_result = False


class TestSparseIsCoalescedAPI6(TestSparseIsCoalescedAPI):
    def setUp(self):
        coo_indices = [[0, 0, 1, 2], [0, 1, 1, 2], [0, 1, 1, 2]]
        coo_values = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
        self.dtype = "complex64"
        self.coo_tensor = paddle.sparse.sparse_coo_tensor(
            coo_indices, coo_values, dtype=self.dtype
        ).coalesce()
        self.expected_result = True


class TestSparseIsCoalescedAPI7(TestSparseIsCoalescedAPI):
    def setUp(self):
        coo_indices = [[0, 0, 1, 2], [0, 1, 1, 2], [1, 0, 1, 2]]
        coo_values = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
        self.dtype = "complex128"
        self.coo_tensor = paddle.sparse.sparse_coo_tensor(
            coo_indices, coo_values, dtype=self.dtype
        )
        self.expected_result = False


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_float16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the float16",
)
class TestSparseIsCoalescedFP16API(TestSparseIsCoalescedAPI):
    def setUp(self):
        self.dtype = "float16"
        coo_indices = [[0, 0, 0, 1], [0, 0, 1, 2]]
        coo_values = paddle.to_tensor([1.0, 2.0, 3.0, 4.0])
        self.coo_tensor = paddle.sparse.sparse_coo_tensor(
            coo_indices, coo_values, dtype=self.dtype
        ).coalesce()
        self.expected_result = True


class TestSparseIsCoalescedAPIStatic(unittest.TestCase):
    def setUp(self):
        self.dtype = "float32"
        self.coo_indices = np.array([[0, 0, 0, 1], [0, 0, 1, 2]]).astype(
            'int64'
        )
        self.coo_values = np.array([1.0, 2.0, 3.0, 4.0]).astype(self.dtype)
        self.coo_shape = [2, 3]
        self.expected_result = False

    def test_is_coalesced(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            coo_indices = paddle.static.data(
                name='coo_indices',
                shape=self.coo_indices.shape,
                dtype='int64',
            )
            coo_values = paddle.static.data(
                name='coo_values',
                shape=self.coo_indices.shape,
                dtype=self.dtype,
            )
            coo = paddle.sparse.sparse_coo_tensor(
                coo_indices,
                coo_values,
                shape=self.coo_shape,
                dtype=self.dtype,
            )

            exe = paddle.static.Executor()
            exe.run(
                feed={
                    'coo_indices': self.coo_indices,
                    'coo_values': self.coo_values,
                }
            )
            self.assertEqual(coo.is_coalesced(), self.expected_result)
        paddle.disable_static()


class TestSparseIsCoalescedAPIStatic1(TestSparseIsCoalescedAPIStatic):
    def setUp(self):
        self.dtype = "float64"
        self.coo_indices = np.array([[0, 1, 0, 1], [0, 0, 1, 2]]).astype(
            'int64'
        )
        self.coo_values = np.array([1.0, 2.0, 3.0, 4.0]).astype(self.dtype)
        self.coo_shape = [2, 3]
        self.expected_result = False


class TestSparseIsCoalescedAPIStatic2(TestSparseIsCoalescedAPIStatic):
    def setUp(self):
        self.dtype = "float64"
        self.coo_indices = np.array([[0, 0, 0, 1], [0, 2, 1, 2]]).astype(
            'int64'
        )
        self.coo_values = np.array([1.0, 2.0, 3.0, 4.0]).astype(self.dtype)
        self.coo_shape = [2, 3]
        self.expected_result = False


class TestSparseIsCoalescedAPIStatic3(TestSparseIsCoalescedAPIStatic):
    def setUp(self):
        self.dtype = "int16"
        self.coo_indices = np.array([[0, 1, 0, 1], [0, 0, 1, 2]]).astype(
            'int64'
        )
        self.coo_values = np.array([1.0, 2.0, 3.0, 4.0]).astype(self.dtype)
        self.coo_shape = [2, 3]
        self.expected_result = False


class TestSparseIsCoalescedAPIStatic4(TestSparseIsCoalescedAPIStatic):
    def setUp(self):
        self.dtype = "int32"
        self.coo_indices = np.array([[0, 0, 0, 1], [0, 2, 1, 2]]).astype(
            'int64'
        )
        self.coo_values = np.array([1.0, 2.0, 3.0, 4.0]).astype(self.dtype)
        self.coo_shape = [2, 3]
        self.expected_result = False


class TestSparseIsCoalescedAPIStatic5(TestSparseIsCoalescedAPIStatic):
    def setUp(self):
        self.dtype = "int64"
        self.coo_indices = np.array([[0, 1, 0, 1], [0, 0, 1, 2]]).astype(
            'int64'
        )
        self.coo_values = np.array([1.0, 2.0, 3.0, 4.0]).astype(self.dtype)
        self.coo_shape = [2, 3]
        self.expected_result = False


class TestSparseIsCoalescedAPIStatic6(TestSparseIsCoalescedAPIStatic):
    def setUp(self):
        self.dtype = "uint8"
        self.coo_indices = np.array([[0, 0, 0, 1], [0, 2, 1, 2]]).astype(
            'int64'
        )
        self.coo_values = np.array([1.0, 2.0, 3.0, 4.0]).astype(self.dtype)
        self.coo_shape = [2, 3]
        self.expected_result = False


class TestSparseIsCoalescedAPIStatic7(TestSparseIsCoalescedAPIStatic):
    def setUp(self):
        self.dtype = "complex64"
        self.coo_indices = np.array([[0, 1, 0, 1], [0, 0, 1, 2]]).astype(
            'int64'
        )
        self.coo_values = np.array([1.0, 2.0, 3.0, 4.0]).astype(self.dtype)
        self.coo_shape = [2, 3]
        self.expected_result = False


class TestSparseIsCoalescedAPIStatic8(TestSparseIsCoalescedAPIStatic):
    def setUp(self):
        self.dtype = "complex128"
        self.coo_indices = np.array([[0, 0, 0, 1], [0, 2, 1, 2]]).astype(
            'int64'
        )
        self.coo_values = np.array([1.0, 2.0, 3.0, 4.0]).astype(self.dtype)
        self.coo_shape = [2, 3]
        self.expected_result = False


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_float16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the float16",
)
class TestSparseIsCoalescedAPIStaticFP16(TestSparseIsCoalescedAPIStatic):
    def setUp(self):
        self.dtype = "float16"
        self.coo_indices = np.array([[0, 0, 0, 1], [0, 2, 1, 2]]).astype(
            'int64'
        )
        self.coo_values = np.array([1.0, 2.0, 3.0, 4.0]).astype(self.dtype)
        self.coo_shape = [2, 3]
        self.expected_result = False


class TestSparseIsCoalescedError(unittest.TestCase):
    def setUp(self):
        self.place = paddle.CPUPlace()

    def test_invalid_tensor_dygraph(self):
        with self.assertRaises(TypeError):
            x = paddle.to_tensor([[1.0, 0.0, 3.0, 0.0, 2.0]]).to_sparse_csr()
            x.is_coalesced()

    def test_invalid_tensor_static_graph(self):
        paddle.enable_static()
        with self.assertRaises(TypeError):
            x_np = np.array([[1.0, 3.0, 2.0]])
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('x', [1], dtype=x_np.dtype)
                x.is_coalesced()
                exe = paddle.static.Executor(self.place)
                exe.run(feed={'x': x_np})
        paddle.disable_static()


if __name__ == "__main__":
    unittest.main()
