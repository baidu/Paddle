# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest

import paddle
from paddle.framework import core


def api_wrapper(
    attn, x, mask, new_mask, keep_first_token=True, keep_order=False
):
    return paddle._C_ops.fused_token_prune(
        attn, x, mask, new_mask, keep_first_token, keep_order
    )


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestFusedTokenPruneOp(OpTest):
    def setDtype(self):
        self.dtype = np.float32

    def setInOuts(self):
        attn = [[1, 2], [3, 4]]
        attn = np.array(attn, dtype=self.dtype)
        attn = np.expand_dims(attn, axis=0)
        self.attn = np.expand_dims(
            attn, axis=0
        )  # [1,1,2,2] bsz = 1, nd_head=1, max_seq_len=2
        mask = [[1, 1], [-1, -1]]
        mask = np.array(mask, dtype=self.dtype)
        mask = np.expand_dims(mask, axis=0)
        self.mask = np.expand_dims(mask, axis=0)  # same as attn
        x = [[1, 2, 3], [4, 5, 6]]
        x = np.array(x, dtype=self.dtype)
        self.x = np.expand_dims(
            x, axis=0
        )  # [1, 2, 3] bsz = 1, max_seq_len=2, c=3
        new_mask = [[1]]
        new_mask = np.array(new_mask, dtype=self.dtype)
        new_mask = np.expand_dims(new_mask, axis=0)
        self.new_mask = np.expand_dims(new_mask, axis=0)  # [1, 1, 1, 1]

        out_slimmedx_py = [[[1, 2, 3]]]
        self.out_slimmedx_py = np.array(out_slimmedx_py, dtype=self.dtype)

        out_cls_inds_py = [[0]]
        self.out_cls_inds_py = np.array(out_cls_inds_py, dtype='int64')

    def setUp(self):
        self.op_type = 'fused_token_prune'
        self.python_api = api_wrapper
        self.python_out_sig = ["SlimmedX", "CLSInds"]
        self.setDtype()
        self.setInOuts()
        self.inputs = {
            'Attn': self.attn,
            'Mask': self.mask,
            'X': self.x,
            'NewMask': self.new_mask,
        }

        self.outputs = {
            'SlimmedX': self.out_slimmedx_py,
            'CLSInds': self.out_cls_inds_py,
        }

    def test_check_output(self):
        self.check_output_with_place(core.CUDAPlace(0))


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestFusedTokenPruneOpFloat64(TestFusedTokenPruneOp):
    def setDtype(self):
        self.dtype = np.float64


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestFusedTokenPruneOp2(TestFusedTokenPruneOp):
    def setInOuts(self):
        attn = [
            [
                [[1, 2, 3, 4], [4, 3, 2, 1], [5, 9, 5, 4], [9, 6, 5, 4]],
                [[8, 5, 2, 0], [1, 0, 2, 3], [2, 2, 3, 2], [7, 4, 1, 8]],
            ]
        ]
        self.attn = np.array(
            attn, dtype=self.dtype
        )  # [1,2,4,4] bsz = 1, nd_head=2, max_seq_len=4
        mask = [
            [
                [
                    [-1, -1, -1, 1],
                    [-1, -1, 1, 1],
                    [-1, -1, 1, 1],
                    [-1, -1, 1, 1],
                ],
                [
                    [-1, -1, 1, 1],
                    [-1, -1, 1, 1],
                    [-1, -1, 1, 1],
                    [-1, -1, 1, 1],
                ],
            ]
        ]
        self.mask = np.array(mask, dtype=self.dtype)  # same as attn
        x = [
            [[1.1, 1.1, 1.1], [2.2, 2.2, 2.2], [3.3, 3.3, 3.3], [4.4, 4.4, 4.4]]
        ]
        self.x = np.array(
            x, dtype=self.dtype
        )  # [1, 4, 3] bsz = 1, max_seq_len=4, c=3
        self.new_mask = np.random.rand(1, 2, 2, 2).astype(
            self.dtype
        )  # [1, 2, 2, 2]

        out_slimmedx_py = [[[1.1, 1.1, 1.1], [4.4, 4.4, 4.4]]]  # [1, 2, 3]
        self.out_slimmedx_py = np.array(out_slimmedx_py, dtype=self.dtype)

        out_cls_inds_py = [[0, 3]]
        self.out_cls_inds_py = np.array(out_cls_inds_py, dtype='int64')


if __name__ == "__main__":
    unittest.main()
