# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fused attention from triton tutorial.
Modified from the original implementation
 - https://github.com/openai/triton/blob/main/python/tutorials/06-fused-attention.py
===============

This is a Triton implementation of the Flash Attention algorithm
(see: Dao et al., https://arxiv.org/pdf/2205.14135v2.pdf; Rabe and Staats https://arxiv.org/pdf/2112.05682v2.pdf)
"""

import torch
import triton
import triton.language as tl


# yapf: disable
@triton.jit
def fused_attention_kernel(
    Out, L, M,  # outputs
    Q, K, V,
    sm_scale,
    batch_size, num_heads, seq_len,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    stride_h = BLOCK_DMODEL * seq_len

    # initialize offsets
    # offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = off_hz * stride_h + offs_m[:, None] * BLOCK_DMODEL + offs_d[None, :]
    off_k = off_hz * stride_h + offs_n[None, :] * BLOCK_DMODEL + offs_d[:, None]
    off_v = off_hz * stride_h + offs_n[:, None] * BLOCK_DMODEL + offs_d[None, :]
    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    # initialize pointer to m and l
    m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    
    #q = tl.load(q_ptrs)
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seq_len, other=0.0)
    sm_scale *= 1.44269504
    # loop over k, v and update accumulator
    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        # -- compute qk ----
        #k = tl.load(k_ptrs)
        k = tl.load(k_ptrs, mask=(start_n + offs_n[None, :] < seq_len), other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        # compute new m
        m_curr = tl.maximum(tl.max(qk, 1), m_prev)
        # correct old l
        l_prev *= tl.math.exp2(m_prev - m_curr)
        # attention weights
        p = tl.math.exp2(qk - m_curr[:, None])
        l_curr = tl.sum(p, 1) + l_prev
        # rescale operands of matmuls
        l_rcp = 1. / l_curr
        p *= l_rcp[:, None]
        acc *= (l_prev * l_rcp)[:, None]
        # update acc
        p = p.to(Q.dtype.element_ty)

        #v = tl.load(v_ptrs)
        v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None] < seq_len), other=0.0)

        acc += tl.dot(p, v)
        # update m_i and l_i
        l_prev = l_curr
        m_prev = m_curr
        # update pointers
        k_ptrs += BLOCK_N * BLOCK_DMODEL
        v_ptrs += BLOCK_N * BLOCK_DMODEL
    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    # l_ptrs = L + off_hz * seq_len + offs_m
    # m_ptrs = M + off_hz * seq_len + offs_m
    # tl.store(l_ptrs, l_prev)
    # tl.store(m_ptrs, m_prev)
    # initialize pointers to output
    offs_n = tl.arange(0, BLOCK_DMODEL)
    off_o = off_hz * stride_h + offs_m[:, None] * BLOCK_DMODEL + offs_n[None, :]
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < seq_len)


def fused_attention(q, k, v, sm_scale, o_buf=None, l_buf=None, m_buf=None):
    BLOCK = 128 if q.dtype == torch.float16 else 64
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = torch.empty_like(q) if o_buf is None else o_buf
    grid = (triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1], 1)
    shape = (q.shape[0] * q.shape[1], q.shape[2])
    L = torch.empty(shape, device=q.device, dtype=torch.float32) if l_buf is None else l_buf
    m = torch.empty(shape, device=q.device, dtype=torch.float32) if m_buf is None else m_buf
    num_warps = 4 if Lk <= 64 else 8

    fused_attention_kernel[grid](
        o, L, m,
        q, k, v,
        sm_scale,
        q.shape[0], q.shape[1], q.shape[2],
        # tl.constexpr
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        BLOCK_DMODEL=Lk,
        num_warps=num_warps,
        num_stages=2,
    )

    return o
# yapf: enable