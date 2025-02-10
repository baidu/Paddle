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

import copy
import math
import os
import re
import unittest

import numpy as np

import paddle
import paddle.sparse
from paddle.base import core
from paddle.base.framework import in_pir_mode


def get_cuda_version():
    result = os.popen("nvcc --version").read()
    regex = r'release (\S+),'
    match = re.search(regex, result)
    if match:
        num = str(match.group(1))
        integer, decimal = num.split('.')
        return int(integer) * 1000 + int(float(decimal) * 10)
    else:
        return -1


@unittest.skipIf(
    not core.is_compiled_with_cuda() or get_cuda_version() < 11080,
    "core is not compiled with CUDA and cuda version need larger than or equal to 11.8",
)
class TestSparseAttentionAPI1(unittest.TestCase):
    def setUp(self):
        paddle.seed(0)
        self.batch_size = 16
        self.num_heads = 16
        self.seq_len = 128
        self.head_dim = 16
        self.dtype = 'float64'
        self.use_mask = True

    def test_dygraph(self):
        self.shape = [
            self.batch_size,
            self.num_heads,
            self.seq_len,
            self.head_dim,
        ]
        query = paddle.rand(self.shape, self.dtype)
        key = paddle.rand(self.shape, self.dtype)
        value = paddle.rand(self.shape, self.dtype)

        query.stop_gradient = False
        key.stop_gradient = False
        value.stop_gradient = False

        mask = paddle.nn.functional.dropout(
            paddle.ones([self.seq_len, self.seq_len]),
            mode='downscale_in_infer',
        )
        mask = mask.expand(
            [self.batch_size, self.num_heads, self.seq_len, self.seq_len]
        )
        sp_mask = mask.reshape([-1, self.seq_len, self.seq_len]).to_sparse_csr()

        query_sp = copy.deepcopy(query)
        key_sp = copy.deepcopy(key)
        value_sp = copy.deepcopy(value)

        query_sp.stop_gradient = False
        key_sp.stop_gradient = False
        value_sp.stop_gradient = False

        if self.use_mask:
            kp_mask = paddle.randint(
                0, 2, [self.batch_size, self.seq_len]
            ).astype(self.dtype)
            attn_mask = paddle.randint(
                0, 2, [self.seq_len, self.seq_len]
            ).astype(self.dtype)

            sdd = paddle.matmul(query, key, False, True) / math.sqrt(
                float(self.head_dim)
            )
            sdd = (
                sdd
                + ((mask * kp_mask.unsqueeze([1, 2]) * attn_mask) - 1.0) * 1e9
            )
            softmax = paddle.nn.functional.softmax(sdd)
            output = paddle.matmul(softmax, value)
            output.backward()

            output_sp = paddle.sparse.nn.functional.attention(
                query_sp, key_sp, value_sp, sp_mask, kp_mask, attn_mask
            )
            output_sp.backward()
        else:
            sdd = paddle.matmul(query, key, False, True) / math.sqrt(
                float(self.head_dim)
            )
            sdd = sdd + (mask - 1.0) * 1e9
            softmax = paddle.nn.functional.softmax(sdd)
            output = paddle.matmul(softmax, value)
            output.backward()

            output_sp = paddle.sparse.nn.functional.attention(
                query_sp, key_sp, value_sp, sp_mask
            )
            output_sp.backward()

        np.testing.assert_allclose(
            output_sp.numpy(), output.numpy(), rtol=1e-05
        )
        np.testing.assert_allclose(
            query_sp.grad.numpy(), query.grad.numpy(), rtol=1e-05
        )
        np.testing.assert_allclose(
            key_sp.grad.numpy(), key.grad.numpy(), rtol=1e-05
        )
        np.testing.assert_allclose(
            value_sp.grad.numpy(), value.grad.numpy(), rtol=1e-05
        )


class TestSparseAttentionAPI2(TestSparseAttentionAPI1):
    def setUp(self):
        super().setUp()
        self.batch_size = 16
        self.num_heads = 16
        self.seq_len = 128
        self.head_dim = 32
        self.dtype = 'float64'
        self.use_mask = False


class TestSparseAttentionAPI3(TestSparseAttentionAPI1):
    def setUp(self):
        super().setUp()
        self.batch_size = 16
        self.num_heads = 16
        self.seq_len = 512
        self.head_dim = 16
        self.dtype = 'float64'
        self.use_mask = True


class TestSparseAttentionAPI4(TestSparseAttentionAPI1):
    def setUp(self):
        super().setUp()
        self.batch_size = 16
        self.num_heads = 16
        self.seq_len = 512
        self.head_dim = 32
        self.dtype = 'float64'
        self.use_mask = False


class TestSparseAttentionAPI5(TestSparseAttentionAPI1):
    def setUp(self):
        super().setUp()
        self.batch_size = 16
        self.num_heads = 16
        self.seq_len = 512
        self.head_dim = 64
        self.dtype = 'float64'
        self.use_mask = True


devices = []
if paddle.device.get_device() != "cpu":
    devices.append(paddle.device.get_device())
else:
    devices.append('cpu')


class TestSparseSoftmaxStaticAPI(unittest.TestCase):
    '''
    Test the API paddle.sparse.nn.functional.softmax on some sparse tensors in pir mode in static graph.
    '''

    def check_result_coo(self, x_shape):
        '''
        x_shape: a tensor shape,
        generate a sparse tensor with shape "x_shape" and compute the output of paddle.sparse.nn.functional.softmax.
        compare the output of paddle.sparse.nn.functional.softmax and the output of paddle.nn.functional.Softmax.
        '''
        for device in devices:
            paddle.device.set_device(device)
            x = paddle.rand(x_shape, dtype='float32')
            indices_data, values_data = (
                x.detach().to_sparse_coo(sparse_dim=len(x_shape)).indices(),
                x.detach().to_sparse_coo(sparse_dim=len(x_shape)).values(),
            )

            x.stop_gradient = False
            out = paddle.nn.functional.softmax(x)

            paddle.enable_static()
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                indices = paddle.static.data(
                    name="indices",
                    shape=indices_data.shape,
                    dtype=indices_data.dtype,
                )
                values = paddle.static.data(
                    name="values",
                    shape=values_data.shape,
                    dtype=values_data.dtype,
                )

                sp_x = paddle.sparse.sparse_coo_tensor(
                    indices,
                    values,
                    shape=x.shape,
                    dtype=x.dtype,
                )
                sp_out = paddle.sparse.nn.functional.softmax(sp_x)
                sp_dense_out = sp_out.to_dense()

                sp_exe = paddle.static.Executor()
                sp_fetch = sp_exe.run(
                    feed={
                        "indices": indices_data.numpy(),
                        "values": values_data.numpy(),
                    },
                    fetch_list=[sp_dense_out],
                    return_numpy=True,
                )
                np.testing.assert_allclose(out.numpy(), sp_fetch[0], rtol=1e-05)
                paddle.disable_static()

    def test_softmax_2d(self):
        if in_pir_mode():
            self.check_result_coo([3, 4])

    def test_softmax_3d(self):
        if in_pir_mode():
            self.check_result_coo([3, 4, 5])

    def test_softmax_4d(self):
        if in_pir_mode():
            self.check_result_coo([3, 4, 5, 6])


if __name__ == '__main__':
    unittest.main()
