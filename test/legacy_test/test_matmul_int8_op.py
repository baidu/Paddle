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

import numpy as np
from test_sparse_attention_op import get_cuda_version

import paddle
from paddle.base import core

paddle.disable_static()


# TODO: verify the requirements of CUDA ARCH
@unittest.skipIf(
    not core.is_compiled_with_cuda() or get_cuda_version() < 11060,
    "MatmulInt8 requires CUDA >= 11.6",
)
class TestMatmulInt8(unittest.TestCase):
    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.x_shape = (8, 64)
        self.y_shape = (64, 64)
        self.trans_x = False
        self.trans_y = True

    def setUp(self):
        self.config()
        self.input_a_np = np.random.randint(-127, 127, self.x_shape).astype(
            'int32'
        )
        self.input_b_np = np.random.randint(-127, 127, self.y_shape).astype(
            'int32'
        )
        self.input_a = paddle.to_tensor(self.input_a_np, dtype=self.dtype)
        self.input_b = paddle.to_tensor(self.input_b_np, dtype=self.dtype)

        if self.trans_x:
            if self.input_a_np.ndim == 1:
                self.input_a_np = self.input_a_np.reshape(
                    (self.input_a_np.size,)
                )
            elif self.input_a_np.ndim == 2:
                self.input_a_np = self.input_a_np.T
            else:
                dim = list(range(len(self.input_a_np.shape)))
                dim[-1], dim[len(self.input_a_np.shape) - 2] = (
                    dim[len(self.input_a_np.shape) - 2],
                    dim[-1],
                )
                self.input_a_np = np.transpose(self.input_a_np, tuple(dim))
        if self.trans_y:
            if self.input_b_np.ndim == 1:
                self.input_b_np = self.input_b_np.reshape(
                    (self.input_b_np.size,)
                )
            elif self.input_b_np.ndim == 2:
                self.input_b_np = self.input_b_np.T
            else:
                dim = list(range(len(self.input_b_np.shape)))
                dim[-1], dim[len(self.input_b_np.shape) - 2] = (
                    dim[len(self.input_b_np.shape) - 2],
                    dim[-1],
                )
                self.input_b_np = np.transpose(self.input_b_np, tuple(dim))

    def get_reference_out(self):
        out = np.matmul(self.input_a_np, self.input_b_np)
        return out

    def get_op_out(self):
        out = paddle._C_ops.matmul(
            self.input_a, self.input_b, self.trans_x, self.trans_y
        )
        return out.numpy()

    def test_matmul_int8(self):
        out_real = self.get_op_out()
        out_expect = self.get_reference_out()
        np.testing.assert_allclose(
            out_real, out_expect, rtol=self.rtol, atol=self.atol
        )


class TestMatmulInt8Op2(TestMatmulInt8):
    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.x_shape = (100,)
        self.y_shape = (1, 3, 2, 100)
        self.trans_x = False
        self.trans_y = True


class TestMatmulInt8Op3(TestMatmulInt8):
    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.x_shape = (4,)
        self.y_shape = (1, 1, 4, 100)
        self.trans_x = False
        self.trans_y = False


class TestMatmulInt8Op4(TestMatmulInt8):
    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.x_shape = (100,)
        self.y_shape = (1, 2, 100, 4)
        self.trans_x = False
        self.trans_y = False


class TestMatmulInt8Op5(TestMatmulInt8):
    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.x_shape = (1, 1, 100, 4)
        self.y_shape = (100,)
        self.trans_x = True
        self.trans_y = False


class TestMatmulInt8Op6(TestMatmulInt8):
    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.x_shape = (1, 2, 104, 4)
        self.y_shape = (104,)
        self.trans_x = True
        self.trans_y = False


class TestMatmulInt8Op7(TestMatmulInt8):
    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.x_shape = (1, 2, 4, 100)
        self.y_shape = (100,)
        self.trans_x = False
        self.trans_y = False


class TestMatmulInt8Op8(TestMatmulInt8):
    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.x_shape = (1, 1, 4, 100)
        self.y_shape = (1, 1, 100, 4)
        self.trans_x = False
        self.trans_y = False


class TestMatmulInt8Op9(TestMatmulInt8):
    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.x_shape = (1, 1, 4, 100)
        self.y_shape = (2, 1, 8, 100)
        self.trans_x = False
        self.trans_y = True


class TestMatmulInt8Op10(TestMatmulInt8):
    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.x_shape = (1, 1, 24, 4)
        self.y_shape = (1, 2, 4, 24)
        self.trans_x = False
        self.trans_y = False


class TestMatmulInt8Op11(TestMatmulInt8):
    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.x_shape = (2, 1, 4, 100)
        self.y_shape = (1, 1, 100, 4)
        self.trans_x = False
        self.trans_y = False


class TestMatmulInt8Op12(TestMatmulInt8):
    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.x_shape = (2, 1, 4, 24)
        self.y_shape = (1, 1, 4, 24)
        self.trans_x = True
        self.trans_y = False


class TestMatmulInt8Op13(TestMatmulInt8):
    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.x_shape = (2, 2, 12, 12)
        self.y_shape = (2, 2, 12, 12)
        self.trans_x = True
        self.trans_y = False


class TestMatmulInt8Op14(TestMatmulInt8):
    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.x_shape = (3, 1, 8, 8)
        self.y_shape = (1, 2, 8, 8)
        self.trans_x = True
        self.trans_y = False


class TestMatmulInt8Op15(TestMatmulInt8):
    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.x_shape = (3, 1, 8, 8)
        self.y_shape = (1, 2, 8, 8)
        self.trans_x = False
        self.trans_y = False


class TestMatmulInt8Op16(TestMatmulInt8):
    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.x_shape = 100
        self.y_shape = (1, 2, 2, 100, 4)
        self.trans_x = False
        self.trans_y = False


class TestMatmulInt8Op17(TestMatmulInt8):
    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.x_shape = (2, 4, 100)
        self.y_shape = 100
        self.trans_x = False
        self.trans_y = False


class TestMatmulInt8OpBroadcast1(TestMatmulInt8):
    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.x_shape = (3, 1, 12, 12)
        self.y_shape = (1, 2, 12, 12)
        self.trans_x = True
        self.trans_y = True


class TestMatmulInt8OpBroadcast2(TestMatmulInt8):
    def config(self):
        self.dtype = 'int8'
        self.rtol = 1e-5
        self.atol = 1e-2
        self.x_shape = (3, 1, 12, 12)
        self.y_shape = (1, 2, 12, 12)
        self.trans_x = False
        self.trans_y = True


if __name__ == '__main__':
    unittest.main()
