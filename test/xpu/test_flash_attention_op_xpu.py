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

import unittest

import numpy as np

import paddle
import paddle.nn.functional as F
from paddle.base import core
from paddle.nn.functional.flash_attention import (
    flash_attention,
    flash_attn_unpadded,
)


def get_triangle_upper_mask(x):
    mask = paddle.full_like(x, -1e4)
    mask.stop_gradient = True
    mask = paddle.triu(mask, diagonal=1)
    mask.stop_gradient = True
    return mask


def attention_naive(q, k, v):
    origin_dtype = q.dtype
    assert k.dtype == origin_dtype
    assert v.dtype == origin_dtype
    if q.dtype != paddle.float32:
        q = paddle.cast(q, "float32")
        k = paddle.cast(k, "float32")
        v = paddle.cast(v, "float32")
    # real calculation
    qt = paddle.transpose(q, [0, 2, 1, 3])
    kt = paddle.transpose(k, [0, 2, 1, 3])
    vt = paddle.transpose(v, [0, 2, 1, 3])
    scale = 1.0 / np.sqrt(q.shape[-1])
    s = paddle.matmul(qt, paddle.transpose(kt, [0, 1, 3, 2]))
    s = paddle.scale(s, scale)
    mask = get_triangle_upper_mask(s)
    s = s + mask
    p = F.softmax(s)
    o = paddle.matmul(p, vt)
    o = paddle.cast(o, origin_dtype)
    return paddle.transpose(o, [0, 2, 1, 3])


def is_flashattn_supported():
    xpu_version = core.get_xpu_device_version(0)
    if xpu_version != core.XPUVersion.XPU3:
        return False
    xhpc_version = paddle.version.xpu_xhpc()
    if xhpc_version == 'False':
        return False
    return True


@unittest.skipIf(
    not is_flashattn_supported(), "only available on XPU3 with XHPC"
)
class TestFlashAttentionAPI(unittest.TestCase):
    def setUp(self):
        self.place = paddle.XPUPlace(0)
        self.shape = (1, 128, 2, 32)
        self.shape_v = (1, 128, 2, 18)
        self.dropout = 0.0
        self.causal = True
        self.return_softmax = False

    def test_all(self):
        self.run_case(dtype="float32", tolerance=5e-4, tolerance_dv=5e-4)
        self.run_case(dtype="float16", tolerance=5e-4, tolerance_dv=1e-3)
        self.run_case(dtype="bfloat16", tolerance=6e-3, tolerance_dv=1e-2)
        self.run_case(
            dtype="float32", tolerance=1e-3, tolerance_dv=1e-3, is_mla=True
        )
        self.run_case(
            dtype="float16", tolerance=5e-4, tolerance_dv=1e-3, is_mla=True
        )
        self.run_case(
            dtype="bfloat16", tolerance=6e-3, tolerance_dv=1e-2, is_mla=True
        )
        self.run_unpadded_case(dtype="float16", rtol=5e-3, atol=1e-3)
        self.run_unpadded_case(dtype="bfloat16", rtol=5e-3, atol=1e-3)

    def run_unpadded_case(self, dtype, rtol, atol):
        self.dtype = dtype
        paddle.disable_static()

        query = np.random.random(self.shape)
        q = paddle.to_tensor(
            query, place=self.place, dtype=self.dtype, stop_gradient=False
        )
        q_ = paddle.to_tensor(
            query, place=self.place, dtype=self.dtype, stop_gradient=False
        )

        out_ = attention_naive(q_, q_, q_)

        scale = 1.0 / np.sqrt(q.shape[-1])

        bs = self.shape[0]
        ms = self.shape[1]
        nh = self.shape[2]
        hd = self.shape[3]
        cu_q = paddle.arange(0, (bs + 1) * ms, ms, dtype='int32')

        qq = paddle.reshape(q, [bs * ms, nh, hd])
        out, _ = flash_attn_unpadded(
            qq,
            qq,
            qq,
            cu_q,
            cu_q,
            ms,
            ms,
            scale,
            self.dropout,
            self.causal,
            self.return_softmax,
        )
        out_ = paddle.reshape(out_, [bs * ms, nh, hd])

        np.testing.assert_allclose(out.numpy(), out_, rtol=rtol, atol=atol)

        out.backward()
        out_.backward()

        np.testing.assert_allclose(
            q.grad.numpy(), q_.grad.numpy(), rtol=rtol, atol=atol
        )

    def run_case(self, dtype, tolerance, tolerance_dv, is_mla=False):
        # TODO(houj04) remove debug codes after correctness check
        print(f"Test case shape {self.shape} dtype {dtype}")

        # test dynamic
        paddle.disable_static()

        np.random.seed(2023)
        query = np.random.uniform(-1.0, 1.0, self.shape)
        key = np.random.uniform(-1.0, 1.0, self.shape)
        value = np.random.uniform(
            -1.0, 1.0, self.shape_v if is_mla else self.shape
        )

        q = paddle.to_tensor(
            query, place=self.place, dtype=dtype, stop_gradient=False
        )
        k = paddle.to_tensor(
            key, place=self.place, dtype=dtype, stop_gradient=False
        )
        v = paddle.to_tensor(
            value, place=self.place, dtype=dtype, stop_gradient=False
        )

        q_ = paddle.to_tensor(
            query, place=self.place, dtype=dtype, stop_gradient=False
        )
        k_ = paddle.to_tensor(
            key, place=self.place, dtype=dtype, stop_gradient=False
        )
        v_ = paddle.to_tensor(
            value, place=self.place, dtype=dtype, stop_gradient=False
        )

        out, _ = flash_attention(
            q, k, v, self.dropout, self.causal, self.return_softmax
        )
        # TODO(houj04): use self.causal
        out_ = attention_naive(q_, k_, v_)

        out.backward()
        out_.backward()

        # forward result
        float_out = paddle.cast(out, "float32")
        float_out_ = paddle.cast(out_, "float32")

        # TODO(houj04) remove debug codes after correctness check
        max_diff_forward = np.max(
            np.abs(float_out.numpy() - float_out_.numpy())
        )
        mean_diff_forward = np.mean(
            np.abs(float_out.numpy() - float_out_.numpy())
        )
        print("max_diff_forward:", max_diff_forward)
        print("mean_diff_forward:", mean_diff_forward)

        np.testing.assert_allclose(
            float_out, float_out_, rtol=tolerance, atol=tolerance
        )

        # backward shape
        self.assertEqual(q.grad.shape, q.shape)
        self.assertEqual(q_.grad.shape, q.shape)
        self.assertEqual(k.grad.shape, k.shape)
        self.assertEqual(k_.grad.shape, k.shape)
        self.assertEqual(v.grad.shape, v.shape)
        self.assertEqual(v_.grad.shape, v.shape)

        # backward result
        float_q_grad = paddle.cast(q.grad, "float32")
        float_q_grad_ = paddle.cast(q_.grad, "float32")
        float_k_grad = paddle.cast(k.grad, "float32")
        float_k_grad_ = paddle.cast(k_.grad, "float32")
        float_v_grad = paddle.cast(v.grad, "float32")
        float_v_grad_ = paddle.cast(v_.grad, "float32")

        # TODO(houj04) remove debug codes after correctness check
        max_diff_q_grad = np.max(
            np.abs(float_q_grad.numpy() - float_q_grad_.numpy())
        )
        mean_diff_q_grad = np.mean(
            np.abs(float_q_grad.numpy() - float_q_grad_.numpy())
        )
        print("max_diff_q_grad:", max_diff_q_grad)
        print("mean_diff_q_grad:", mean_diff_q_grad)

        max_diff_k_grad = np.max(
            np.abs(float_k_grad.numpy() - float_k_grad_.numpy())
        )
        mean_diff_k_grad = np.mean(
            np.abs(float_k_grad.numpy() - float_k_grad_.numpy())
        )
        print("max_diff_k_grad:", max_diff_k_grad)
        print("mean_diff_k_grad:", mean_diff_k_grad)

        max_diff_v_grad = np.max(
            np.abs(float_v_grad.numpy() - float_v_grad_.numpy())
        )
        mean_diff_v_grad = np.mean(
            np.abs(float_v_grad.numpy() - float_v_grad_.numpy())
        )
        print("max_diff_v_grad:", max_diff_v_grad)
        print("mean_diff_v_grad:", mean_diff_v_grad)

        np.testing.assert_allclose(
            float_q_grad, float_q_grad_, rtol=tolerance, atol=tolerance
        )
        np.testing.assert_allclose(
            float_k_grad, float_k_grad_, rtol=tolerance, atol=tolerance
        )
        np.testing.assert_allclose(
            float_v_grad, float_v_grad_, rtol=tolerance_dv, atol=tolerance_dv
        )


class TestFlashAttentionAPITest1(TestFlashAttentionAPI):
    def setUp(self):
        self.place = paddle.XPUPlace(0)
        self.shape = (2, 128, 1, 32)
        self.shape_v = (2, 128, 1, 18)
        self.dropout = 0.0
        self.causal = True
        self.return_softmax = False


# class TestFlashAttentionAPITest2(TestFlashAttentionAPI):
#     def setUp(self):
#         self.place = paddle.XPUPlace(0)
#         self.shape = (1, 8192, 5, 128)
#         self.dropout = 0.0
#         self.causal = True
#         self.return_softmax = False


# The following three REAL unit tests are disabled because they take a VERY LONG time to run, although they all pass under XHPC v20240105.

# class TestFlashAttentionAPITestEB(TestFlashAttentionAPI):
#    def setUp(self):
#        self.place = paddle.XPUPlace(0)
#        self.shape = (4, 4096, 4, 128)
#        self.dropout = 0.0
#        self.causal = True
#        self.return_softmax = False

# class TestFlashAttentionAPITestLlama7B(TestFlashAttentionAPI):
#    def setUp(self):
#        self.place = paddle.XPUPlace(0)
#        self.shape = (2, 2048, 16, 128)
#        self.dropout = 0.0
#        self.causal = True
#        self.return_softmax = False

# class TestFlashAttentionAPITestLlama65B(TestFlashAttentionAPI):
#    def setUp(self):
#        self.place = paddle.XPUPlace(0)
#        self.shape = (2, 8192, 8, 128)
#        self.dropout = 0.0
#        self.causal = True
#        self.return_softmax = False


if __name__ == '__main__':
    unittest.main()
