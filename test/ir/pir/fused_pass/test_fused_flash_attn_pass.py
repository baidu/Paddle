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

import os
import re
import unittest

import numpy as np
from pass_test import PassTest

import paddle
from paddle.base import core

np.random.seed(42)
paddle.enable_static()


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


is_sm_supported = (
    core.is_compiled_with_cuda()
    and paddle.device.cuda.get_device_capability()[0] >= 8
    and paddle.device.cuda.get_device_capability()[1] >= 0
)


def is_flashattn_supported():
    if (
        not core.is_compiled_with_cuda()
        or get_cuda_version() < 11040
        or not is_sm_supported
    ):
        return False
    return True


@unittest.skipIf(
    not is_flashattn_supported(),
    "core is not compiled with CUDA and cuda version need larger than or equal to 11.4"
    "and device's compute capability must >= 8.x",
)
class TestFlashAttnPatternQscaleCast(PassTest):
    r"""
         Q          K           V
         |          |           |
     transpose  transpose   transpose
         |          |           |
       scale    transpose       |
         |          |           |
         -- matmul--            |
              |                 |
    mask --- add                |
              |                 |
            cast                |
              |                 |
           softmax              |
              |                 |
             cast               |
              |                 |
              ------matmul------
                      |
                  transpose
                      |
                     out

         Q   K   V   None   mask
         |   |   |     |      |
         ------flash_attn------
                   |
                  out
    """

    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        for bs in [1]:
            for seq_len in [128]:
                for head_dim in [64]:
                    for num_heads in [8]:
                        with paddle.pir_utils.IrGuard():
                            main_prog = paddle.static.Program()
                            start_prog = paddle.static.Program()
                            with paddle.pir.core.program_guard(
                                main_prog, start_prog
                            ):
                                mask_shape = (bs, 1, seq_len, seq_len)
                                Q = paddle.static.data(
                                    name='Q',
                                    shape=[bs, seq_len, num_heads, head_dim],
                                    dtype='float16',
                                )
                                K = paddle.static.data(
                                    name='K',
                                    shape=[bs, seq_len, num_heads, head_dim],
                                    dtype='float16',
                                )
                                V = paddle.static.data(
                                    name='V',
                                    shape=[bs, seq_len, num_heads, head_dim],
                                    dtype='float16',
                                )
                                mask = paddle.static.data(
                                    name='mask',
                                    shape=mask_shape,
                                    dtype='float16',
                                )
                                qt = paddle.transpose(Q, [0, 2, 1, 3])
                                q_scale = paddle.scale(
                                    qt, scale=0.125, bias=0.0
                                )
                                kt = paddle.transpose(K, [0, 2, 1, 3])
                                kt = paddle.transpose(kt, [0, 1, 3, 2])
                                vt = paddle.transpose(V, [0, 2, 1, 3])
                                score = paddle.matmul(q_scale, kt)
                                score = paddle.add(score, mask)
                                cast_out = paddle.cast(score, 'float16')
                                softmax_out = paddle.nn.functional.softmax(
                                    cast_out
                                )
                                softmax_out = paddle.cast(
                                    softmax_out, 'float16'
                                )
                                attention_out = paddle.matmul(softmax_out, vt)
                                attention_out = paddle.transpose(
                                    attention_out, [0, 2, 1, 3]
                                )
                                out = paddle.assign(attention_out)
                                self.pass_attr_list = [
                                    {'fused_flash_attn_pass': {}}
                                ]
                                self.feeds = {
                                    "Q": np.random.random(
                                        (bs, seq_len, num_heads, head_dim)
                                    ).astype("float16"),
                                    "K": np.random.random(
                                        (bs, seq_len, num_heads, head_dim)
                                    ).astype("float16"),
                                    "V": np.random.random(
                                        (bs, seq_len, num_heads, head_dim)
                                    ).astype("float16"),
                                    "mask": np.random.random(mask_shape).astype(
                                        "float16"
                                    ),
                                }
                                self.fetch_list = [out]
                                self.valid_op_map = {
                                    "pd_op.flash_attn": 1,
                                }
                                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct(atol=1e-3, rtol=1e-3)


@unittest.skipIf(
    not is_flashattn_supported(),
    "core is not compiled with CUDA and cuda version need larger than or equal to 11.4"
    "and device's compute capability must >= 8.x",
)
class TestFlashAttnPatternQscaleNoCast(PassTest):
    r"""
         Q          K           V
         |          |           |
     transpose  transpose   transpose
         |          |           |
       scale    transpose       |
         |          |           |
         -- matmul--            |
              |                 |
    mask --- add                |
              |                 |
              |                 |
           softmax              |
              |                 |
              |                 |
              ------matmul------
                      |
                  transpose
                      |
                     out

         Q   K   V   None   mask
         |   |   |     |      |
         ------flash_attn------
                   |
                  out
    """

    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        for bs in [1]:
            for seq_len in [128]:
                for head_dim in [64]:
                    for num_heads in [8]:
                        with paddle.pir_utils.IrGuard():
                            main_prog = paddle.static.Program()
                            start_prog = paddle.static.Program()
                            with paddle.pir.core.program_guard(
                                main_prog, start_prog
                            ):
                                mask_shape = (bs, 1, seq_len, seq_len)
                                Q = paddle.static.data(
                                    name='Q',
                                    shape=[bs, seq_len, num_heads, head_dim],
                                    dtype='float16',
                                )
                                K = paddle.static.data(
                                    name='K',
                                    shape=[bs, seq_len, num_heads, head_dim],
                                    dtype='float16',
                                )
                                V = paddle.static.data(
                                    name='V',
                                    shape=[bs, seq_len, num_heads, head_dim],
                                    dtype='float16',
                                )
                                mask = paddle.static.data(
                                    name='mask',
                                    shape=mask_shape,
                                    dtype='float16',
                                )
                                qt = paddle.transpose(Q, [0, 2, 1, 3])
                                q_scale = paddle.scale(
                                    qt, scale=0.125, bias=0.0
                                )
                                kt = paddle.transpose(K, [0, 2, 1, 3])
                                kt = paddle.transpose(kt, [0, 1, 3, 2])
                                vt = paddle.transpose(V, [0, 2, 1, 3])
                                score = paddle.matmul(q_scale, kt)
                                score = paddle.add(score, mask)
                                softmax_out = paddle.nn.functional.softmax(
                                    score
                                )
                                attention_out = paddle.matmul(softmax_out, vt)
                                attention_out = paddle.transpose(
                                    attention_out, [0, 2, 1, 3]
                                )
                                out = paddle.assign(attention_out)
                                self.pass_attr_list = [
                                    {'fused_flash_attn_pass': {}}
                                ]
                                self.feeds = {
                                    "Q": np.random.random(
                                        (bs, seq_len, num_heads, head_dim)
                                    ).astype("float16"),
                                    "K": np.random.random(
                                        (bs, seq_len, num_heads, head_dim)
                                    ).astype("float16"),
                                    "V": np.random.random(
                                        (bs, seq_len, num_heads, head_dim)
                                    ).astype("float16"),
                                    "mask": np.random.random(mask_shape).astype(
                                        "float16"
                                    ),
                                }
                                self.fetch_list = [out]
                                self.valid_op_map = {
                                    "pd_op.flash_attn": 1,
                                }
                                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct(atol=1e-3, rtol=1e-3)


@unittest.skipIf(
    not is_flashattn_supported(),
    "core is not compiled with CUDA and cuda version need larger than or equal to 11.4"
    "and device's compute capability must >= 8.x",
)
class TestFlashAttnPatternOutscaleCast(PassTest):
    r"""
         Q          K           V
         |          |           |
     transpose  transpose   transpose
         |          |           |
         |      transpose       |
         |          |           |
         -- matmul--            |
              |                 |
            scale               |
              |                 |
    mask --- add                |
              |                 |
            cast                |
              |                 |
           softmax              |
              |                 |
             cast               |
              |                 |
              ------matmul------
                      |
                  transpose
                      |
                     out

         Q   K   V   None   mask
         |   |   |     |      |
         ------flash_attn------
                   |
                  out
    """

    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        for bs in [1]:
            for seq_len in [128]:
                for head_dim in [64]:
                    for num_heads in [8]:
                        with paddle.pir_utils.IrGuard():
                            main_prog = paddle.static.Program()
                            start_prog = paddle.static.Program()
                            with paddle.pir.core.program_guard(
                                main_prog, start_prog
                            ):
                                mask_shape = (bs, 1, seq_len, seq_len)
                                Q = paddle.static.data(
                                    name='Q',
                                    shape=[bs, seq_len, num_heads, head_dim],
                                    dtype='float16',
                                )
                                K = paddle.static.data(
                                    name='K',
                                    shape=[bs, seq_len, num_heads, head_dim],
                                    dtype='float16',
                                )
                                V = paddle.static.data(
                                    name='V',
                                    shape=[bs, seq_len, num_heads, head_dim],
                                    dtype='float16',
                                )
                                mask = paddle.static.data(
                                    name='mask',
                                    shape=mask_shape,
                                    dtype='float16',
                                )
                                qt = paddle.transpose(Q, [0, 2, 1, 3])
                                kt = paddle.transpose(K, [0, 2, 1, 3])
                                kt = paddle.transpose(kt, [0, 1, 3, 2])
                                vt = paddle.transpose(V, [0, 2, 1, 3])

                                score = paddle.matmul(qt, kt)
                                score_scale = paddle.scale(
                                    score, scale=0.125, bias=0.0
                                )
                                score = paddle.add(score_scale, mask)
                                cast_out = paddle.cast(score, 'float16')
                                softmax_out = paddle.nn.functional.softmax(
                                    cast_out
                                )
                                softmax_out = paddle.cast(
                                    softmax_out, 'float16'
                                )
                                attention_out = paddle.matmul(softmax_out, vt)
                                attention_out = paddle.transpose(
                                    attention_out, [0, 2, 1, 3]
                                )
                                out = paddle.assign(attention_out)
                                self.pass_attr_list = [
                                    {'fused_flash_attn_pass': {}}
                                ]
                                self.feeds = {
                                    "Q": np.random.random(
                                        (bs, seq_len, num_heads, head_dim)
                                    ).astype("float16"),
                                    "K": np.random.random(
                                        (bs, seq_len, num_heads, head_dim)
                                    ).astype("float16"),
                                    "V": np.random.random(
                                        (bs, seq_len, num_heads, head_dim)
                                    ).astype("float16"),
                                    "mask": np.random.random(mask_shape).astype(
                                        "float16"
                                    ),
                                }
                                self.fetch_list = [out]
                                self.valid_op_map = {
                                    "pd_op.flash_attn": 1,
                                }
                                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct(atol=1e-3, rtol=1e-3)


@unittest.skipIf(
    not is_flashattn_supported(),
    "core is not compiled with CUDA and cuda version need larger than or equal to 11.4"
    "and device's compute capability must >= 8.x",
)
class TestFlashAttnPatternOutscaleNoCast(PassTest):
    r"""
         Q          K           V
         |          |           |
     transpose  transpose   transpose
         |          |           |
         |    transpose         |
         |          |           |
         -- matmul--            |
              |                 |
            scale               |
              |                 |
    mask --- add                |
              |                 |
              |                 |
           softmax              |
              |                 |
              |                 |
              ------matmul------
                      |
                  transpose
                      |
                     out

         Q   K   V   None   mask
         |   |   |     |      |
         ------flash_attn------
                   |
                  out
    """

    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        for bs in [1]:
            for seq_len in [128]:
                for head_dim in [64]:
                    for num_heads in [8]:
                        with paddle.pir_utils.IrGuard():
                            main_prog = paddle.static.Program()
                            start_prog = paddle.static.Program()
                            with paddle.pir.core.program_guard(
                                main_prog, start_prog
                            ):
                                mask_shape = (bs, 1, seq_len, seq_len)
                                Q = paddle.static.data(
                                    name='Q',
                                    shape=[bs, seq_len, num_heads, head_dim],
                                    dtype='float16',
                                )
                                K = paddle.static.data(
                                    name='K',
                                    shape=[bs, seq_len, num_heads, head_dim],
                                    dtype='float16',
                                )
                                V = paddle.static.data(
                                    name='V',
                                    shape=[bs, seq_len, num_heads, head_dim],
                                    dtype='float16',
                                )
                                mask = paddle.static.data(
                                    name='mask',
                                    shape=mask_shape,
                                    dtype='float16',
                                )
                                qt = paddle.transpose(Q, [0, 2, 1, 3])
                                kt = paddle.transpose(K, [0, 2, 1, 3])
                                kt = paddle.transpose(kt, [0, 1, 3, 2])
                                vt = paddle.transpose(V, [0, 2, 1, 3])

                                score = paddle.matmul(qt, kt)
                                score_scale = paddle.scale(
                                    score, scale=0.125, bias=0.0
                                )
                                score = paddle.add(score_scale, mask)
                                softmax_out = paddle.nn.functional.softmax(
                                    score
                                )
                                attention_out = paddle.matmul(softmax_out, vt)
                                attention_out = paddle.transpose(
                                    attention_out, [0, 2, 1, 3]
                                )
                                out = paddle.assign(attention_out)
                                self.pass_attr_list = [
                                    {'fused_flash_attn_pass': {}}
                                ]
                                self.feeds = {
                                    "Q": np.random.random(
                                        (bs, seq_len, num_heads, head_dim)
                                    ).astype("float16"),
                                    "K": np.random.random(
                                        (bs, seq_len, num_heads, head_dim)
                                    ).astype("float16"),
                                    "V": np.random.random(
                                        (bs, seq_len, num_heads, head_dim)
                                    ).astype("float16"),
                                    "mask": np.random.random(mask_shape).astype(
                                        "float16"
                                    ),
                                }
                                self.fetch_list = [out]
                                self.valid_op_map = {
                                    "pd_op.flash_attn": 1,
                                }
                                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct(atol=1e-3, rtol=1e-3)


@unittest.skipIf(
    not is_flashattn_supported(),
    "core is not compiled with CUDA and cuda version need larger than or equal to 11.4"
    "and device's compute capability must >= 8.x",
)
class TestFlashAttnPatternOutscaleCastNoMask(PassTest):
    r"""
        Q          K           V
        |          |           |
    transpose  transpose   transpose
        |          |           |
        -- matmul--            |
             |                 |
           scale               |
             |                 |
           cast                |
             |                 |
          softmax              |
             |                 |
            cast               |
             |                 |
             ------matmul------
                     |
                 transpose
                     |
                    out

        Q   K   V   None   None
        |   |   |     |      |
        ------flash_attn------
                  |
                 out
    """

    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        for bs in [1]:
            for seq_len in [128]:
                for head_dim in [64]:
                    for num_heads in [8]:
                        with paddle.pir_utils.IrGuard():
                            main_prog = paddle.static.Program()
                            start_prog = paddle.static.Program()
                            with paddle.pir.core.program_guard(
                                main_prog, start_prog
                            ):
                                Q = paddle.static.data(
                                    name='Q',
                                    shape=[bs, seq_len, num_heads, head_dim],
                                    dtype='float16',
                                )
                                K = paddle.static.data(
                                    name='K',
                                    shape=[bs, seq_len, num_heads, head_dim],
                                    dtype='float16',
                                )
                                V = paddle.static.data(
                                    name='V',
                                    shape=[bs, seq_len, num_heads, head_dim],
                                    dtype='float16',
                                )
                                qt = paddle.transpose(Q, [0, 2, 1, 3])
                                kt = paddle.transpose(K, [0, 2, 1, 3])
                                vt = paddle.transpose(V, [0, 2, 1, 3])

                                score = paddle.matmul(qt, kt, transpose_y=True)
                                score_scale = paddle.scale(
                                    score, scale=0.125, bias=0.0
                                )
                                cast_out = paddle.cast(score_scale, 'float16')
                                softmax_out = paddle.nn.functional.softmax(
                                    cast_out
                                )
                                softmax_out = paddle.cast(
                                    softmax_out, 'float16'
                                )
                                attention_out = paddle.matmul(softmax_out, vt)
                                attention_out = paddle.transpose(
                                    attention_out, [0, 2, 1, 3]
                                )
                                out = paddle.assign(attention_out)
                                self.pass_attr_list = [
                                    {'fused_flash_attn_pass': {}}
                                ]
                                self.feeds = {
                                    "Q": np.random.random(
                                        (bs, seq_len, num_heads, head_dim)
                                    ).astype("float16"),
                                    "K": np.random.random(
                                        (bs, seq_len, num_heads, head_dim)
                                    ).astype("float16"),
                                    "V": np.random.random(
                                        (bs, seq_len, num_heads, head_dim)
                                    ).astype("float16"),
                                }
                                self.fetch_list = [out]
                                self.valid_op_map = {
                                    "pd_op.flash_attn": 1,
                                }
                                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct(atol=1e-3, rtol=1e-3)


@unittest.skipIf(
    not is_flashattn_supported(),
    "core is not compiled with CUDA and cuda version need larger than or equal to 11.4"
    "and device's compute capability must >= 8.x",
)
class TestFlashAttnPatternOutscaleNoCastNoMask(PassTest):
    r"""
        Q          K           V
        |          |           |
    transpose  transpose   transpose
        |          |           |
        -- matmul--            |
             |                 |
           scale               |
             |                 |
          softmax              |
             |                 |
             |                 |
             ------matmul------
                     |
                 transpose
                     |
                    out

        Q   K   V   None   None
        |   |   |     |      |
        ------flash_attn------
                  |
                 out
    """

    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        for bs in [1]:
            for seq_len in [128]:
                for head_dim in [64]:
                    for num_heads in [8]:
                        with paddle.pir_utils.IrGuard():
                            main_prog = paddle.static.Program()
                            start_prog = paddle.static.Program()
                            with paddle.pir.core.program_guard(
                                main_prog, start_prog
                            ):
                                Q = paddle.static.data(
                                    name='Q',
                                    shape=[bs, seq_len, num_heads, head_dim],
                                    dtype='float16',
                                )
                                K = paddle.static.data(
                                    name='K',
                                    shape=[bs, seq_len, num_heads, head_dim],
                                    dtype='float16',
                                )
                                V = paddle.static.data(
                                    name='V',
                                    shape=[bs, seq_len, num_heads, head_dim],
                                    dtype='float16',
                                )
                                qt = paddle.transpose(Q, [0, 2, 1, 3])
                                kt = paddle.transpose(K, [0, 2, 1, 3])
                                vt = paddle.transpose(V, [0, 2, 1, 3])

                                score = paddle.matmul(qt, kt, transpose_y=True)
                                score_scale = paddle.scale(
                                    score, scale=0.125, bias=0.0
                                )
                                softmax_out = paddle.nn.functional.softmax(
                                    score_scale
                                )
                                attention_out = paddle.matmul(softmax_out, vt)
                                attention_out = paddle.transpose(
                                    attention_out, [0, 2, 1, 3]
                                )
                                out = paddle.assign(attention_out)
                                self.pass_attr_list = [
                                    {'fused_flash_attn_pass': {}}
                                ]
                                self.feeds = {
                                    "Q": np.random.random(
                                        (bs, seq_len, num_heads, head_dim)
                                    ).astype("float16"),
                                    "K": np.random.random(
                                        (bs, seq_len, num_heads, head_dim)
                                    ).astype("float16"),
                                    "V": np.random.random(
                                        (bs, seq_len, num_heads, head_dim)
                                    ).astype("float16"),
                                }
                                self.fetch_list = [out]
                                self.valid_op_map = {
                                    "pd_op.flash_attn": 1,
                                }
                                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct(atol=1e-3, rtol=1e-3)


@unittest.skipIf(
    not is_flashattn_supported(),
    "core is not compiled with CUDA and cuda version need larger than or equal to 11.4"
    "and device's compute capability must >= 8.x",
)
class TestTransposeSliceFlashAttnPattern(PassTest):
    r"""
                   transpose
                       |
            -----------+----------
            |          |           |
          slice       slice      slice
            |          |           |
            Q          K           V
            |          |           |
            |       transpose      |
            |          |           |
            -- matmul--            |
                 |                 |
               scale               |
                 |                 |
       mask --- add                |
                 |                 |
              softmax              |
                 |                 |
                 ------matmul------
                         |
                     transpose
                         |
                        out

              transpose
                  |
            ------+------
            |     |     |
          slice slice slice
            |     |     |
            Q     K     V              mask
            |     |     |               |
    transpose transpose transpose       |
            |     |     |               |
            -------flash_attn------------
                      |
                     out
    """

    def is_program_valid(self, program=None):
        return True

    def build_ir_program(self):
        for bs in [1]:
            for seq_len in [128]:
                for head_dim in [64]:
                    for num_heads in [8]:
                        with paddle.pir_utils.IrGuard():
                            main_prog = paddle.static.Program()
                            start_prog = paddle.static.Program()
                            with paddle.pir.core.program_guard(
                                main_prog, start_prog
                            ):
                                x = paddle.static.data(
                                    name='x',
                                    shape=[bs, seq_len, 3, num_heads, head_dim],
                                    dtype='float16',
                                )
                                mask_shape = (bs, 1, seq_len, seq_len)
                                mask = paddle.static.data(
                                    name='mask',
                                    shape=mask_shape,
                                    dtype='float16',
                                )
                                xt = paddle.transpose(x, [2, 0, 3, 1, 4])
                                q = xt[0, :, :, :, :]
                                k = xt[1, :, :, :, :]
                                v = xt[2, :, :, :, :]
                                kt = paddle.transpose(k, [0, 1, 3, 2])

                                score = paddle.matmul(q, kt)
                                score_scale = paddle.scale(
                                    score, scale=0.125, bias=0.0
                                )
                                score_add = paddle.add(score_scale, mask)
                                softmax_out = paddle.nn.functional.softmax(
                                    score_add
                                )
                                attention_out = paddle.matmul(softmax_out, v)
                                attention_out = paddle.transpose(
                                    attention_out, [0, 2, 1, 3]
                                )
                                out = paddle.assign(attention_out)
                                self.pass_attr_list = [
                                    {'fused_flash_attn_pass': {}}
                                ]
                                self.feeds = {
                                    "x": np.random.random(
                                        (bs, seq_len, 3, num_heads, head_dim)
                                    ).astype("float16"),
                                    "mask": np.random.random(mask_shape).astype(
                                        "float16"
                                    ),
                                }
                                self.fetch_list = [out]
                                self.valid_op_map = {
                                    "pd_op.flash_attn": 1,
                                }
                                return [main_prog, start_prog]

    def sample_program(self):
        yield self.build_ir_program(), False

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct(atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
