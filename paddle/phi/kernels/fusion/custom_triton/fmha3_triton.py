"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)
Credits: OpenAI kernel team

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import triton
import triton.language as tl
import fmha3_triton_util
# We don't run auto-tuning everytime to keep the tutorial fast. Uncommenting
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
# @triton.autotune(
#    configs=[
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=8),
#        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
#        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32}, num_stages=3, num_warps=8),
#        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32}, num_stages=3, num_warps=4),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=3, num_warps=4),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=4, num_warps=4),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=4, num_warps=4),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=7, num_warps=8),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=7, num_warps=8),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=6, num_warps=8),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=5, num_warps=8),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=4, num_warps=8),
#        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=6, num_warps=4),
#    ],
#    key=['N_CTX'],
# )


@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, Out,  #
            #   stride_qz, stride_qh, stride_qm, stride_qk,  #
            #   stride_kz, stride_kh, stride_kn, stride_kk,  #
            #   stride_vz, stride_vh, stride_vk, stride_vn,  #
            #   stride_oz, stride_oh, stride_om, stride_on,  #
              Z, H, S, D,#
              N_CTX: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_DMODEL: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              ):
    stride_qz = H * S * D
    # stride_kz = stride_qz
    # stride_vz = stride_qz
    # stride_oz = stride_qz

    stride_qh = S * D
    # stride_kh = stride_qh
    # stride_vh = stride_qh
    # stride_oh = stride_qh

    # stride_qm = D
    # stride_kn = D
    # stride_vk = D
    # stride_om = D

    # stride_qk = 1
    # stride_kk = 1
    # stride_vn = 1
    # stride_on = 1
    # stride_qh = stride_kh = stride_vh = stride_oh = S * D
    # stride_qm = stride_kn = stride_vk = stride_om = D
    # stride_qk = stride_kk = stride_vn = stride_on = 1

    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        # shape=(N_CTX, BLOCK_DMODEL),
        shape=(S, BLOCK_DMODEL),
        strides=(D, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        # shape=(N_CTX, BLOCK_DMODEL),
        shape=(S, BLOCK_DMODEL),
        strides=(D, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        # shape=(BLOCK_DMODEL, N_CTX),
        shape=(BLOCK_DMODEL, S),
        strides=(1, D),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        # shape=(N_CTX, BLOCK_DMODEL),
        shape=(S, BLOCK_DMODEL),
        strides=(D, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    # q = tl.load(Q_block_ptr, boundary_check=(0, ))
    q = tl.load(Q_block_ptr)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    #if STAGE & 1:
    acc, l_i, m_i = fmha3_triton_util._attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                    start_m, qk_scale,  S,#
                                    BLOCK_M, BLOCK_DMODEL, BLOCK_N,  #
                                    1, offs_m, offs_n, N_CTX  #
                                    )
    
    # stage 2: on-band
    #if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        #tl.debug_barrier()
    acc, l_i, m_i = fmha3_triton_util._attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                    start_m, qk_scale,  S,#
                                    BLOCK_M, BLOCK_DMODEL, BLOCK_N,  #
                                    2, offs_m, offs_n, N_CTX  #
                                    )
    # if STAGE & 4:
    #     #
    #     acc, l_i, m_i = fmha3_triton_util._attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
    #                                     start_m, qk_scale,  #
    #                                     BLOCK_M, BLOCK_DMODEL, BLOCK_N,  #
    #                                     4, offs_m, offs_n, N_CTX  #
    #                                     )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    # m_ptrs = M + off_hz * N_CTX + offs_m
    # 这个先注释掉，后面再改
    # tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(0,))


