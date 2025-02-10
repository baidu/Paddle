#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import numpy as np

import paddle
import paddle.nn.functional as F
from paddle import _C_ops, in_dynamic_mode
from paddle.base.framework import in_dynamic_or_pir_mode
from paddle.base.layer_helper import LayerHelper
from paddle.base.wrapped_decorator import signature_safe_contextmanager
from paddle.device.cuda import get_device_capability

g_enable_math = None
g_enable_flash = None
g_enable_mem_efficient = None

if TYPE_CHECKING:
    from collections.abc import Generator

    from paddle import Tensor


def _get_arch_info():
    # Get SMVersion from device.
    cuda_version = paddle.version.cuda()
    if (
        cuda_version is not None and cuda_version != 'False'
    ) or paddle.is_compiled_with_rocm():
        major, minor = get_device_capability()
        arch = int(major * 10 + minor)
        return arch
    else:
        raise ValueError(
            "Paddle is not compiled with CUDA, we cannot get SMVersion from device, please try to compile Paddle with CUDA"
        )


def check_flash_head_dim_constraints(query, dropout_p=0.0):
    arch = _get_arch_info()
    is_sm86_to_sm89 = 86 <= arch <= 89

    if not is_sm86_to_sm89:
        return True

    head_dim = query.shape[-1]
    requires_grad = not query.stop_gradient

    if not requires_grad:
        return True

    is_head_dim_gt192 = head_dim > 192
    is_head_dim_lte224 = head_dim <= 224
    is_dropout = dropout_p > 0.0

    cond1 = is_head_dim_gt192 and is_head_dim_lte224
    cond2 = head_dim > 224 and is_dropout

    if cond1 or cond2:
        return False
    return True


def check_flash_causal_non_square_seqlens(query, key, is_causal=False):
    if not is_causal:
        return True

    seqlen_q = query.shape[-3]
    seqlen_k = key.shape[-3]

    if seqlen_q != seqlen_k:
        return False
    return True


def check_dtypes_low_precision(query, debug=False):
    arch = _get_arch_info()
    dtype = query.dtype

    if arch >= 80:
        supported_dtypes = [paddle.float16, paddle.bfloat16]
    else:
        supported_dtypes = [paddle.float16]

    return dtype in supported_dtypes


def can_use_flash_attn(query, key, attn_mask, dropout, is_causal) -> bool:
    # sdpa flash check
    # step1 check tensor place on cuda
    # step2 check tensor shape, flash attn only support shape == 4
    # step3 check head_dim <= 256
    # step4 check arch_info > sm80
    # step5 check specify sm head dim constraint
    # step6 check causal qk
    # step7 check sm dtype support
    if "gpu" not in paddle.get_device():
        return False
    if query.ndim != 4:
        return False
    if query.shape[-1] >= 256:
        return False
    if _get_arch_info() < 80:
        return False
    if not check_flash_head_dim_constraints(query, dropout):
        return False
    if not check_flash_causal_non_square_seqlens(query, key, is_causal):
        return False
    if not check_dtypes_low_precision(query):
        return False
    return True


def can_use_efficient(query) -> bool:
    # sdpa efficient check
    # step1 check tensor place on cuda
    # step2 check arch_info in [sm50, sm90]
    # step3 check tensor shape, mem efficient only support shape == 4
    if "gpu" not in paddle.get_device():
        return False
    if _get_arch_info() < 50 and _get_arch_info() > 90:
        return False
    if query.ndim != 4:
        return False
    return True


@signature_safe_contextmanager
def sdp_kernel(
    enable_math: bool = False,
    enable_flash: bool = True,
    enable_mem_efficient: bool = True,
) -> Generator[None, None, None]:
    r"""
    With the sdp_kernel context manager, different algorithm implementations can
    be selected for scaled_dot_product_attention.
    """
    global g_enable_math, g_enable_flash, g_enable_mem_efficient
    original_enable_math = g_enable_math
    original_enable_flash = g_enable_math
    original_enable_mem_efficient = g_enable_mem_efficient

    g_enable_math = enable_math
    g_enable_flash = enable_flash
    g_enable_mem_efficient = enable_mem_efficient
    try:
        yield
    finally:
        g_enable_math = original_enable_math
        g_enable_flash = original_enable_flash
        g_enable_mem_efficient = original_enable_mem_efficient


# special for XPU device
def get_triangle_upper_mask(x: Tensor) -> Tensor:
    mask = paddle.full_like(x, -1e4)
    mask.stop_gradient = True
    mask = paddle.triu(mask, diagonal=1)
    mask.stop_gradient = True
    return mask


@overload
def _math_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor,
    dropout_rate: float = ...,
    causal: bool = ...,
    return_softmax: Literal[False] = ...,
    training: bool = ...,
) -> tuple[Tensor, None]: ...


@overload
def _math_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor,
    dropout_rate: float = ...,
    causal: bool = ...,
    return_softmax: Literal[True] = ...,
    training: bool = ...,
) -> tuple[Tensor, Tensor]: ...


@overload
def _math_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor,
    dropout_rate: float = ...,
    causal: bool = ...,
    return_softmax: bool = ...,
    training: bool = ...,
) -> tuple[Tensor, Tensor | None]: ...


def _math_attention(
    query,
    key,
    value,
    mask=None,
    dropout_rate=0.0,
    causal=False,
    return_softmax=False,
    training=True,
):
    r"""
    This is a basic implementation of scaled dot product attention composed of
    combinations of fundamental components.
    """
    head_dim = query.shape[-1]
    query = paddle.transpose(query, [0, 2, 1, 3])
    key = paddle.transpose(key, [0, 2, 1, 3])
    value = paddle.transpose(value, [0, 2, 1, 3])
    product = paddle.matmul(x=query * (head_dim**-0.5), y=key, transpose_y=True)

    if mask is not None:
        product = product + mask

    if not causal:
        weights = F.softmax(product)
    else:
        # special for XPU device
        place = paddle.get_device()
        if "xpu" in place:
            # softmax_mask_fuse_upper_triangle is not supported on XPU, use plain implementation
            mask = get_triangle_upper_mask(product)
            product = product + mask
            weights = F.softmax(product)
        else:
            weights = paddle.incubate.softmax_mask_fuse_upper_triangle(product)
    if dropout_rate > 0.0:
        weights = F.dropout(
            weights, dropout_rate, training=training, mode="upscale_in_train"
        )

    out = paddle.matmul(weights, value)
    out = paddle.transpose(out, [0, 2, 1, 3])
    return out, weights if return_softmax else None


def _select_sdp_cuda(head_dim: int) -> str:

    if head_dim <= 256:
        return "flash_attn"
    else:
        return "mem_efficient"


def _select_sdp(head_dim: int) -> str:
    r"""
    There are currently three different implementation options available for
    scaled dot product attention, and the chosen approach depends on whether it
    is determined by the sdp_kernel configuration or specified through input values.
    """
    place = paddle.get_device()
    device = place.split(":")[0]

    if "xpu" in place:
        return "flash_attn"

    # not use sdp_kernel
    if g_enable_flash is None:
        if "gpu" not in place and not paddle.is_compiled_with_custom_device(
            device
        ):
            return "math"
        else:
            return _select_sdp_cuda(head_dim)

    if (
        g_enable_math is False
        and g_enable_flash is False
        and g_enable_mem_efficient is False
    ):
        raise AssertionError(
            "No available backend for scaled_dot_product_attention was found."
        )

    if g_enable_math is True:
        if g_enable_flash is False and g_enable_mem_efficient is False:
            return "math"
        if "gpu" not in place:
            return "math"
    if g_enable_flash is True and g_enable_mem_efficient is True:
        return _select_sdp_cuda(head_dim)
    if g_enable_flash is True:
        return "flash_attn"
    return "mem_efficient"


def _select_sdp_for_sdpa(query, key, attn_mask, dropout, is_causal) -> str:
    r"""
    this select sdpa is alignment for torch version
    """
    place = paddle.get_device()
    if "xpu" in place:
        return "flash_attn"

    # not use sdp_kernel
    if (
        g_enable_flash is None
        and g_enable_math is None
        and g_enable_mem_efficient is None
    ):
        # test flash attn usage
        use_flash = can_use_flash_attn(
            query, key, attn_mask, dropout, is_causal
        )
        use_efficient = can_use_efficient(query)
        use_math = True
        if use_flash:
            return "flash_attn"
        elif use_efficient:
            return "mem_efficient"
        elif use_math:
            return "math"

    if (
        g_enable_math is False
        and g_enable_flash is False
        and g_enable_mem_efficient is False
    ):
        raise AssertionError(
            "No available backend for scaled_dot_product_attention was found."
        )

    if g_enable_math is True:
        if g_enable_flash is False and g_enable_mem_efficient is False:
            return "math"
        if "gpu" not in place:
            return "math"
    if g_enable_flash is True and g_enable_mem_efficient is True:
        return _select_sdp_cuda(query.shape[-1])
    if g_enable_flash is True:
        return "flash_attn"
    return "mem_efficient"


@overload
def flash_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout: float = ...,
    causal: bool = ...,
    return_softmax: Literal[False] = ...,
    *,
    fixed_seed_offset: Tensor | None = ...,
    rng_name: str = ...,
    training: bool = ...,
    name: str | None = ...,
) -> tuple[Tensor, None]: ...


@overload
def flash_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout: float = ...,
    causal: bool = ...,
    return_softmax: Literal[True] = ...,
    *,
    fixed_seed_offset: Tensor | None = ...,
    rng_name: str = ...,
    training: bool = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor]: ...


@overload
def flash_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout: float = ...,
    causal: bool = ...,
    return_softmax: bool = ...,
    *,
    fixed_seed_offset: Tensor | None = ...,
    rng_name: str = ...,
    training: bool = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor | None]: ...


def flash_attention(
    query,
    key,
    value,
    dropout=0.0,
    causal=False,
    return_softmax=False,
    *,
    fixed_seed_offset=None,
    rng_name="",
    training=True,
    name=None,
):
    r"""
    The equation is:

    .. math::

        result=softmax(\frac{ Q * K^T }{\sqrt{d}}) * V

    where : ``Q``, ``K``, and ``V`` represent the three input parameters of the attention module.
    The dimensions of the three parameters are the same.
    ``d`` represents the size of the last dimension of the three parameters.

    Warning:
        This API is only support inputs with dtype float16 and bfloat16.

    Args:
        query(Tensor): The query tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seq_len, num_heads, head_dim].
                        The dtype can be float61 or bfloat16.
        key(Tensor): The key tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seq_len, num_heads, head_dim].
                        The dtype can be float61 or bfloat16.
        value(Tensor): The value tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seq_len, num_heads, head_dim].
                        The dtype can be float61 or bfloat16.
        dropout(float): The dropout ratio.
        causal(bool): Whether enable causal mode.
        return_softmax(bool): Whether to return softmax.
        fixed_seed_offset(Tensor|None, optional): With fixed seed, offset for dropout mask.
        training(bool): Whether it is in the training phase.
        rng_name(str): The name to select Generator.
        name(str|None, optional): The default value is None. Normally there is no need for user
                        to set this property. For more information, please refer to
                        :ref:`api_guide_Name`.

    Returns:
        out(Tensor): The attention tensor.
                    4-D tensor with shape: [batch_size, seq_len, num_heads, head_dim].
                    The dtype can be float16 or bfloat16.
        softmax(Tensor): The softmax tensor. None if return_softmax is False.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> paddle.seed(2023)
            >>> q = paddle.rand((1, 128, 2, 16))

            >>> output = paddle.nn.functional.flash_attention.flash_attention(q, q, q, 0.9, False, False)
            >>> print(output)
            (Tensor(shape=[1, 128, 2, 16], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[[0.34992966, 0.34456208, 0.45826620, ..., 0.39883569,
                0.42132431, 0.39157745],
               [0.76687670, 0.65837246, 0.69117945, ..., 0.82817286,
                0.76690865, 0.71485823]],
              ...,
              [[0.71662450, 0.57275224, 0.57053083, ..., 0.48108247,
                0.53336465, 0.54540104],
               [0.59137970, 0.51350880, 0.50449550, ..., 0.38860250,
                0.40526697, 0.60541755]]]]), None)

    """
    head_dim = query.shape[3]
    sdp_func_name = _select_sdp(head_dim)

    if sdp_func_name == "flash_attn":
        if in_dynamic_or_pir_mode():
            (result_attention, result_softmax, _, _) = _C_ops.flash_attn(
                query,
                key,
                value,
                fixed_seed_offset,
                None,
                dropout,
                causal,
                return_softmax,
                not training,
                rng_name,
            )
            return result_attention, result_softmax if return_softmax else None

        helper = LayerHelper('flash_attn', **locals())
        dtype = helper.input_dtype(input_param_name='q')
        out = helper.create_variable_for_type_inference(dtype)
        softmax = helper.create_variable_for_type_inference(dtype)
        softmax_lse = helper.create_variable_for_type_inference(paddle.float32)
        seed_offset = helper.create_variable_for_type_inference(paddle.int64)
        inputs = {
            'q': query,
            'k': key,
            'v': value,
            'fixed_seed_offset': fixed_seed_offset,
        }
        outputs = {
            'out': out,
            'softmax': softmax,
            'softmax_lse': softmax_lse,
            'seed_offset': seed_offset,
        }
        helper.append_op(
            type='flash_attn',
            inputs=inputs,
            outputs=outputs,
            attrs={
                'dropout': dropout,
                'causal': causal,
                'return_softmax': return_softmax,
                'is_test': not training,
                'rng_name': rng_name,
            },
        )
        return out, softmax if return_softmax else None
    else:
        if sdp_func_name == "mem_efficient":
            from paddle.incubate.nn.memory_efficient_attention import (
                memory_efficient_attention,
            )

            output = memory_efficient_attention(
                query,
                key,
                value,
                attn_bias=None,
                p=dropout,
                scale=None,
                training=training,
            )
            return output, None
        else:
            return _math_attention(
                query,
                key,
                value,
                dropout_rate=dropout,
                causal=causal,
                return_softmax=return_softmax,
                training=training,
            )


@overload
def flash_attn_qkvpacked(
    qkv: Tensor,
    dropout: float = ...,
    causal: bool = ...,
    return_softmax: Literal[False] = ...,
    *,
    fixed_seed_offset: Tensor | None = ...,
    rng_name: str = ...,
    training: bool = ...,
    name: str | None = ...,
) -> tuple[Tensor, None]: ...


@overload
def flash_attn_qkvpacked(
    qkv: Tensor,
    dropout: float = ...,
    causal: bool = ...,
    return_softmax: Literal[True] = ...,
    *,
    fixed_seed_offset: Tensor | None = ...,
    rng_name: str = ...,
    training: bool = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor]: ...


@overload
def flash_attn_qkvpacked(
    qkv: Tensor,
    dropout: float = ...,
    causal: bool = ...,
    return_softmax: bool = ...,
    *,
    fixed_seed_offset: Tensor | None = ...,
    rng_name: str = ...,
    training: bool = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor | None]: ...


def flash_attn_qkvpacked(
    qkv,
    dropout=0.0,
    causal=False,
    return_softmax=False,
    *,
    fixed_seed_offset=None,
    rng_name="",
    training=True,
    name=None,
):
    r"""
    The equation is:

    .. math::

        result=softmax(\frac{ Q * K^T }{\sqrt{d}}) * V

    where : ``Q``, ``K``, and ``V`` represent the three input parameters of the attention module.
    The dimensions of the three parameters are the same.
    ``d`` represents the size of the last dimension of the three parameters.

    Warning:
        This API only supports inputs with dtype float16 and bfloat16.
        Don't call this API if flash_attn is not supported.

    Args:
        qkv(Tensor): The query/key/value packed tensor in the Attention module.
                        5-D tensor with shape:
                        [batchsize, seqlen , num_heads/num_heads_k + 2, num_heads_k, head_dim].
                        The dtype can be float16 or bfloat16.
        dropout(float): The dropout ratio.
        causal(bool): Whether enable causal mode.
        return_softmax(bool): Whether to return softmax.
        fixed_seed_offset(Tensor|None, optional): With fixed seed, offset for dropout mask.
        training(bool): Whether it is in the training phase.
        rng_name(str): The name to select Generator.
        name(str|None, optional): The default value is None. Normally there is no need for user
                        to set this property. For more information, please refer to
                        :ref:`api_guide_Name`.

    Returns:
        - out(Tensor). The attention tensor. 4-D tensor with shape: [batch_size, seq_len, num_heads, head_dim]. The dtype can be float16 or bfloat16.
        - softmax(Tensor). The softmax tensor. None if return_softmax is False.

    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP('flash_attn need A100 compile')
            >>> import paddle

            >>> paddle.seed(2023)
            >>> q = paddle.rand((1, 128, 2, 16))
            >>> qkv = paddle.stack([q, q, q], axis=2)
            >>> output = paddle.nn.functional.flash_attn_qkvpacked(qkv, 0.9, False, False)
            >>> print(output)
            (Tensor(shape=[1, 128, 2, 16], dtype=float32, place=Place(cpu), stop_gradient=True,
            [[[[0.34992966, 0.34456208, 0.45826620, ..., 0.39883569,
                0.42132431, 0.39157745],
               [0.76687670, 0.65837246, 0.69117945, ..., 0.82817286,
                0.76690865, 0.71485823]],
              ...,
              [[0.71662450, 0.57275224, 0.57053083, ..., 0.48108247,
                0.53336465, 0.54540104],
               [0.59137970, 0.51350880, 0.50449550, ..., 0.38860250,
                0.40526697, 0.60541755]]]]), None)
            >>> # doctest: -SKIP

    """
    head_dim = qkv.shape[-1]
    sdp_func_name = _select_sdp(head_dim)

    if sdp_func_name == "flash_attn":
        if in_dynamic_or_pir_mode():
            (
                result_attention,
                result_softmax,
                _,
                _,
            ) = _C_ops.flash_attn_qkvpacked(
                qkv,
                fixed_seed_offset,
                None,
                dropout,
                causal,
                return_softmax,
                not training,
                rng_name,
            )
            return result_attention, result_softmax if return_softmax else None

        helper = LayerHelper('flash_attn_qkvpacked', **locals())
        dtype = helper.input_dtype(input_param_name='qkv')
        out = helper.create_variable_for_type_inference(dtype)
        softmax = helper.create_variable_for_type_inference(dtype)
        softmax_lse = helper.create_variable_for_type_inference(paddle.float32)
        seed_offset = helper.create_variable_for_type_inference(paddle.int64)
        inputs = {
            'qkv': qkv,
            'fixed_seed_offset': fixed_seed_offset,
        }
        outputs = {
            'out': out,
            'softmax': softmax,
            'softmax_lse': softmax_lse,
            'seed_offset': seed_offset,
        }
        helper.append_op(
            type='flash_attn_qkvpacked',
            inputs=inputs,
            outputs=outputs,
            attrs={
                'dropout': dropout,
                'causal': causal,
                'return_softmax': return_softmax,
                'is_test': not training,
                'rng_name': rng_name,
            },
        )
        return out, softmax if return_softmax else None
    else:
        # don't call qkvpacked if not using flash_attn
        query = qkv[:, :, :-2].reshape([0, 0, -1, qkv.shape[-1]])
        key = qkv[:, :, -2]
        value = qkv[:, :, -1]
        if sdp_func_name == "mem_efficient":
            from paddle.incubate.nn.memory_efficient_attention import (
                memory_efficient_attention,
            )

            output = memory_efficient_attention(
                query,
                key,
                value,
                attn_bias=None,
                p=dropout,
                scale=None,
                training=training,
            )
            return output, None
        else:
            return _math_attention(
                query,
                key,
                value,
                dropout_rate=dropout,
                causal=causal,
                return_softmax=return_softmax,
                training=training,
            )


@overload
def flash_attn_unpadded(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: float,
    dropout: float = ...,
    causal: bool = ...,
    return_softmax: Literal[False] = ...,
    fixed_seed_offset: Tensor | None = ...,
    rng_name: str = ...,
    training: bool = ...,
    name: str | None = ...,
) -> tuple[Tensor, None]: ...


@overload
def flash_attn_unpadded(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: float,
    dropout: float = ...,
    causal: bool = ...,
    return_softmax: Literal[True] = ...,
    fixed_seed_offset: Tensor | None = ...,
    rng_name: str = ...,
    training: bool = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor]: ...


@overload
def flash_attn_unpadded(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: float,
    dropout: float = ...,
    causal: bool = ...,
    return_softmax: bool = ...,
    fixed_seed_offset: Tensor | None = ...,
    rng_name: str = ...,
    training: bool = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor | None]: ...


def flash_attn_unpadded(
    query,
    key,
    value,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    scale,
    dropout=0.0,
    causal=False,
    return_softmax=False,
    fixed_seed_offset=None,
    rng_name='',
    training=True,
    name=None,
):
    r"""
    The equation is:

    .. math::

        result=softmax(\frac{ Q * K^T }{\sqrt{d}}) * V

    where : ``Q``, ``K``, and ``V`` represent the three input parameters of the attention module.
    The dimensions of the three parameters are the same.
    ``d`` represents the size of the last dimension of the three parameters.

    Warning:
        This API is only support inputs with dtype float16 and bfloat16.

    Args:
        query(Tensor): The query tensor in the Attention module.
                        3-D tensor with shape:
                        [total_seq_len, num_heads, head_dim].
                        The dtype can be float61 or bfloat16.
        key(Tensor): The key tensor in the Attention module.
                        3-D tensor with shape:
                        [total_seq_len, num_heads, head_dim].
                        The dtype can be float61 or bfloat16.
        value(Tensor): The value tensor in the Attention module.
                        3-D tensor with shape:
                        [total_seq_len, num_heads, head_dim].
                        The dtype can be float61 or bfloat16.
        cu_seqlens_q(Tensor): The cumulative sequence lengths of the sequences in the batch,
                        used to index query.
        cu_seqlens_k(Tensor): The cumulative sequence lengths of the sequences in the batch,
                        used to index key and value.
        max_seqlen_q(int): Maximum sequence length of query in the batch.
        max_seqlen_k(int): Maximum sequence length of key/value in the batch.
        scale(float): The scaling of QK^T before applying softmax.
        dropout(float, optional): The dropout ratio.
        causal(bool, optional): Whether enable causal mode.
        return_softmax(bool, optional): Whether to return softmax.
        fixed_seed_offset(Tensor|None, optional): With fixed seed, offset for dropout mask.
        rng_name(str, optional): The name to select Generator.
        training(bool, optional): Whether it is in the training phase.
        name(str|None, optional): The default value is None. Normally there is no need for user
                        to set this property. For more information, please refer to
                        :ref:`api_guide_Name`.

    Returns:
        out(Tensor): The attention tensor.
                    3-D tensor with shape: [total_seq_len, num_heads, head_dim].
                    The dtype can be float16 or bfloat16.
        softmax(Tensor): The softmax tensor. None if return_softmax is False.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.seed(2023)
            >>> q = paddle.rand((2, 128, 8, 16), dtype='float16')
            >>> cu = paddle.arange(0, 384, 128, dtype='int32')
            >>> qq = paddle.reshape(q, [256, 8, 16])
            >>> output = paddle.nn.functional.flash_attention.flash_attn_unpadded(qq, qq, qq, cu, cu, 128, 128, 0.25, 0.0, False, False)

    """
    if in_dynamic_mode():
        (
            result_attention,
            result_softmax,
        ) = _C_ops.flash_attn_unpadded(
            query,
            key,
            value,
            cu_seqlens_q,
            cu_seqlens_k,
            fixed_seed_offset,
            None,
            max_seqlen_q,
            max_seqlen_k,
            scale,
            dropout,
            causal,
            return_softmax,
            not training,
            rng_name,
        )
        return result_attention, result_softmax if return_softmax else None

    helper = LayerHelper('flash_attn_unpadded', **locals())
    dtype = helper.input_dtype(input_param_name='q')
    out = helper.create_variable_for_type_inference(dtype)
    softmax = helper.create_variable_for_type_inference(dtype)
    softmax_lse = helper.create_variable_for_type_inference(paddle.float32)
    seed_offset = helper.create_variable_for_type_inference(paddle.int64)
    inputs = {
        'q': query,
        'k': key,
        'v': value,
        'cu_seqlens_q': cu_seqlens_q,
        'cu_seqlens_k': cu_seqlens_k,
        'fixed_seed_offset': fixed_seed_offset,
    }
    outputs = {
        'out': out,
        'softmax': softmax,
        'softmax_lse': softmax_lse,
        'seed_offset': seed_offset,
    }
    helper.append_op(
        type='flash_attn_unpadded',
        inputs=inputs,
        outputs=outputs,
        attrs={
            'max_seqlen_q': max_seqlen_q,
            'max_seqlen_k': max_seqlen_k,
            'scale': scale,
            'dropout': dropout,
            'causal': causal,
            'return_softmax': return_softmax,
            'is_test': not training,
            'rng_name': rng_name,
        },
    )
    return out, softmax if return_softmax else None


@overload
def flash_attn_varlen_qkvpacked(
    qkv: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: float,
    dropout: float = ...,
    causal: bool = ...,
    return_softmax: Literal[False] = ...,
    fixed_seed_offset: Tensor | None = ...,
    rng_name: str = ...,
    varlen_padded: bool = ...,
    training: bool = ...,
    name: str | None = ...,
) -> tuple[Tensor, None]: ...


@overload
def flash_attn_varlen_qkvpacked(
    qkv: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: float,
    dropout: float = ...,
    causal: bool = ...,
    return_softmax: Literal[True] = ...,
    fixed_seed_offset: Tensor | None = ...,
    rng_name: str = ...,
    varlen_padded: bool = ...,
    training: bool = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor]: ...


@overload
def flash_attn_varlen_qkvpacked(
    qkv: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: float,
    dropout: float = ...,
    causal: bool = ...,
    return_softmax: bool = ...,
    fixed_seed_offset: Tensor | None = ...,
    rng_name: str = ...,
    varlen_padded: bool = ...,
    training: bool = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor | None]: ...


def flash_attn_varlen_qkvpacked(
    qkv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    scale,
    dropout=0.0,
    causal=False,
    return_softmax=False,
    fixed_seed_offset=None,
    rng_name="",
    varlen_padded=True,
    training=True,
    name=None,
):
    r"""
    The equation is:

    .. math::

        result=softmax(\frac{ Q * K^T }{\sqrt{d}}) * V

    where : ``Q``, ``K``, and ``V`` represent the three input parameters of the attention module.
    The dimensions of the three parameters are the same.
    ``d`` represents the size of the last dimension of the three parameters.

    Warning:
        This API only supports inputs with dtype float16 and bfloat16.

    Args:
        qkv(Tensor): The padded query/key/value packed tensor in the Attention module. The padding part won't be computed
                        4-D tensor with shape:
                        [total_seq_len, num_heads/num_heads_k + 2, num_heads_k, head_dim].
                        The dtype can be float16 or bfloat16.
        cu_seqlens_q(Tensor): The cumulative sequence lengths of the sequences in the batch,
                        used to index query.
        cu_seqlens_k(Tensor): The cumulative sequence lengths of the sequences in the batch,
                        used to index key and value.
        max_seqlen_q(int): Maximum sequence length of query in the batch. Note it's the padding length, not the max actual seqlen
        max_seqlen_k(int): Maximum sequence length of key/value in the batch.
        scale(float): The scaling of QK^T before applying softmax.
        dropout(float, optional): The dropout ratio.
        causal(bool, optional): Whether enable causal mode.
        return_softmax(bool, optional): Whether to return softmax.
        fixed_seed_offset(Tensor|None, optional): With fixed seed, offset for dropout mask.
        rng_name(str, optional): The name to select Generator.
        training(bool, optional): Whether it is in the training phase.
        name(str|None, optional): The default value is None. Normally there is no need for user
                        to set this property. For more information, please refer to
                        :ref:`api_guide_Name`.

    Returns:
        - out(Tensor). The attention tensor. The tensor is padded by zeros. 3-D tensor with shape: [total_seq_len, num_heads, head_dim]. The dtype can be float16 or bfloat16.
        - softmax(Tensor). The softmax tensor. None if return_softmax is False.

    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP('flash_attn need A100 compile')
            >>> import paddle
            >>> paddle.seed(2023)
            >>> q = paddle.rand((2, 128, 8, 16), dtype='float16')
            >>> cu = paddle.arange(0, 384, 128, dtype='int32')
            >>> qq = paddle.reshape(q, [256, 8, 16])
            >>> qkv = paddle.stack([qq, qq, qq], axis=2)
            >>> output = paddle.nn.functional.flash_attn_varlen_qkvpacked(qkv, cu, cu, 128, 128, 0.25, 0.0, False, False)
            >>> # doctest: -SKIP

    """
    if in_dynamic_mode():
        (
            result_attention,
            result_softmax,
        ) = _C_ops.flash_attn_varlen_qkvpacked(
            qkv,
            cu_seqlens_q,
            cu_seqlens_k,
            fixed_seed_offset,
            None,
            max_seqlen_q,
            max_seqlen_k,
            scale,
            dropout,
            causal,
            return_softmax,
            not training,
            rng_name,
            varlen_padded,
        )
        return result_attention, result_softmax if return_softmax else None

    helper = LayerHelper('flash_attn_varlen_qkvpacked', **locals())
    dtype = helper.input_dtype(input_param_name='qkv')
    out = helper.create_variable_for_type_inference(dtype)
    softmax = helper.create_variable_for_type_inference(dtype)
    softmax_lse = helper.create_variable_for_type_inference(paddle.float32)
    seed_offset = helper.create_variable_for_type_inference(paddle.int64)
    inputs = {
        'qkv': qkv,
        'cu_seqlens_q': cu_seqlens_q,
        'cu_seqlens_k': cu_seqlens_k,
        'fixed_seed_offset': fixed_seed_offset,
    }
    outputs = {
        'out': out,
        'softmax': softmax,
        'softmax_lse': softmax_lse,
        'seed_offset': seed_offset,
    }
    helper.append_op(
        type='flash_attn_varlen_qkvpacked',
        inputs=inputs,
        outputs=outputs,
        attrs={
            'max_seqlen_q': max_seqlen_q,
            'max_seqlen_k': max_seqlen_k,
            'scale': scale,
            'dropout': dropout,
            'causal': causal,
            'return_softmax': return_softmax,
            'is_test': not training,
            'rng_name': rng_name,
        },
    )
    return out, softmax if return_softmax else None


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    training: bool = True,
    name: str | None = None,
) -> Tensor:
    r"""
    The equation is:

    .. math::

        result=softmax(\frac{ Q * K^T }{\sqrt{d}}) * V

    where : ``Q``, ``K``, and ``V`` represent the three input parameters of the attention module.
    The dimensions of the three parameters are the same.
    ``d`` represents the size of the last dimension of the three parameters.

    Warning:
        This API only supports inputs with dtype float16 and bfloat16.

    Args:
        query(Tensor): The query tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seq_len, num_heads, head_dim].
                        The dtype can be float16 or bfloat16.
        key(Tensor): The key tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seq_len, num_heads, head_dim].
                        The dtype can be float16 or bfloat16.
        value(Tensor): The value tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seq_len, num_heads, head_dim].
                        The dtype can be float16 or bfloat16.
        attn_mask(Tensor, optional): A float mask of the same type as query,
                        key, value that is added to the attention score.
        dropout_p(float, optional): The dropout ratio.
        is_causal(bool, optional): Whether enable causal mode.
        training(bool, optional): Whether it is in the training phase.
        name(str|None, optional): The default value is None. Normally there is no need for user
                        to set this property. For more information, please refer to
                        :ref:`api_guide_Name`.

    Returns:
        out(Tensor): The attention tensor.
                    4-D tensor with shape: [batch_size, seq_len, num_heads, head_dim].
                    The dtype can be float16 or bfloat16.

    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP('bfloat need V100 compile')
            >>> import paddle
            >>> q = paddle.rand((1, 128, 2, 16), dtype=paddle.bfloat16)
            >>> output = paddle.nn.functional.scaled_dot_product_attention(q, q, q, None, 0.9, False)
            >>> print(output)
            >>> # doctest: -SKIP
    """

    if attn_mask is None:
        # downgraded to ordinary flash attention implementation
        out, _ = flash_attention(query, key, value, dropout_p, is_causal)
        return out
    else:
        head_dim = query.shape[3]
        sdp_func_name = _select_sdp_for_sdpa(
            query, key, attn_mask, dropout_p, is_causal
        )
        if sdp_func_name == "flash_attn":
            if in_dynamic_or_pir_mode():
                fixed_seed_offset = None
                return_softmax = False
                rng_name = ""
                out, _, _, _ = _C_ops.flash_attn(
                    query,
                    key,
                    value,
                    fixed_seed_offset,
                    attn_mask,
                    dropout_p,
                    is_causal,
                    return_softmax,
                    not training,
                    rng_name,
                )
                return out
            else:
                helper = LayerHelper('flash_attn', **locals())
                dtype = helper.input_dtype(input_param_name='q')
                out = helper.create_variable_for_type_inference(dtype)
                softmax = helper.create_variable_for_type_inference(dtype)
                softmax_lse = helper.create_variable_for_type_inference(
                    paddle.float32
                )
                seed_offset = helper.create_variable_for_type_inference(
                    paddle.int64
                )
                inputs = {
                    'q': query,
                    'k': key,
                    'v': value,
                    'attn_mask': attn_mask,
                }
                outputs = {
                    'out': out,
                    'softmax': softmax,
                    'softmax_lse': softmax_lse,
                    'seed_offset': seed_offset,
                }
                helper.append_op(
                    type='flash_attn',
                    inputs=inputs,
                    outputs=outputs,
                    attrs={
                        'dropout': dropout_p,
                        'causal': is_causal,
                        'return_softmax': False,
                        'is_test': not training,
                        'rng_name': '',
                    },
                )
                return out
        elif sdp_func_name == "mem_efficient":
            from paddle.incubate.nn.functional.variable_length_memory_efficient_attention import (
                variable_length_memory_efficient_attention,
            )

            seq_lens = paddle.to_tensor(
                [query.shape[1]] * query.shape[0], dtype='int32'
            )

            scale = 1.0 / np.sqrt(query.shape[-1])

            query = query.transpose([0, 2, 1, 3])
            key = key.transpose([0, 2, 1, 3])
            value = value.transpose([0, 2, 1, 3])

            output = variable_length_memory_efficient_attention(
                query, key, value, seq_lens, seq_lens, attn_mask, scale
            )

            output = output.transpose([0, 2, 1, 3])

            return output
        elif sdp_func_name == "math":
            return _math_attention(
                query,
                key,
                value,
                attn_mask,
                dropout_p,
                is_causal,
                False,
                training,
            )[0]


def flashmask_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    startend_row_indices: Tensor | None = None,
    *,
    dropout: float = 0.0,
    causal: bool = False,
    window_size: int | tuple | None = None,
    return_softmax_lse: bool = False,
    return_seed_offset: bool = False,
    fixed_seed_offset: Tensor | None = None,
    rng_name: str = "",
    training: bool = True,
    name: str | None = None,
):
    r"""
    FlashMask: Official Implementation

    This module provides the official implementation of the FlashMask algorithm as described in the paper. For more details, please refer to the paper available at: https://arxiv.org/abs/2410.01359.

    The core equation utilized in FlashMask is as follows:

    .. math::

        \text{result} = \text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d}} + M\right) \cdot V

    In this equation:

        - ``Q``, ``K``, and ``V`` are the input tensors to the attention module.
        - All these tensors share the same dimensions.
        - ``d`` denotes the size of the last dimension of these tensors.
        - ``M`` represents the column-wise sparse mask introduced by FlashMask.

    Args:
        query (Tensor):  The query tensor in the attention module.
            A 4-D tensor with shape [batch_size, q_seq_len, num_heads, head_dim].
            The dtype can be float16 or bfloat16.
        key (Tensor): The key tensor in the attention module.
            A 4-D tensor with shape [batch_size, k_seq_len, k_num_heads, head_dim].
            The dtype can be float16 or bfloat16.
        value (Tensor): The value tensor in the attention module.
            A 4-D tensor with shape [batch_size, k_seq_len, k_num_heads, head_dim].
            The dtype can be float16 or bfloat16.
        startend_row_indices(Tensor):
            A column-wise sparse attention mask row indices tensor.
            A 4-D tensor with shape [batch_size, k_num_heads, k_seq_len, {1, 2, 4}].
            The dtype must be int32. k_num_heads can be 1 or the same as key's num_heads. When num_heads is 1, it will be broadcast to match key's num_heads.
            Depending on the value of the causal parameter, startend_row_indices can take different shapes and meanings.

            - When `causal=True` and the shape is [batch_size, k_num_heads, k_seq_len, 1],
              indicating unidirectional attention. The value represents the starting row index of the left
              lower triangular mask in the dense mask. The value startend_row_indices[..., 0] indicates that elements in the lower left triangle of the attention score matrix starting from the startend_row_indices[..., 0]-th row downwards (inclusive) will be masked.
            - When `causal=True` and the shape is [batch_size, k_num_heads, k_seq_len, 2],
              indicating unidirectional attention. The values represent the starting and ending row indices of
              the left lower triangular mask in the dense mask. The values startend_row_indices[..., 0:2] in startend_row_indices indicate that elements in the lower left triangle of the attention score matrix starting from the startend_row_indices[..., 0]-th row downwards (inclusive) but above the startend_row_indices[..., 1]-th row (exclusive) will be masked.
            - When `causal=False` and the shape is [batch_size, k_num_heads, k_seq_len, 2],
              indicating bidirectional attention. The values represent the starting row index of the left
              lower triangular mask and the ending row index of the right upper triangular mask in the dense mask. The values startend_row_indices[..., 0:2] in startend_row_indices indicate that elements in the lower left triangle of the attention score matrix starting from the startend_row_indices[..., 0]-th row downwards (inclusive) will be masked, and elements in the upper right triangle starting from the startend_row_indices[..., 1]-th row upwards (exclusive) will be masked.
            - When `causal=False` and the shape is [batch_size, k_num_heads, k_seq_len, 4] ,
              indicating bidirectional attention. The values represent the start and end row indices of the
              left lower triangular mask and the start and end row indices of the right upper triangular mask in the dense mask. The values startend_row_indices[..., 0:4] in startend_row_indices indicate that elements in the lower left triangle of the attention score matrix starting from the startend_row_indices[..., 0]-th row downwards (inclusive) but above the startend_row_indices[..., 1] row (exclusive) will be masked, and elements in the upper right triangle starting from the startend_row_indices[..., 2]-th row downwards (inclusive) but above the startend_row_indices[..., 3] row (exclusive) will be masked.

        dropout (float): The dropout ratio. Default is 0.0.
        causal (bool): Whether to enable causal mode. Default is False.
        window_size (int|tuple, optional): Indicates the window size of sliding window local attention.
            If causal mode is enabled, Query at position i will only attend to keys between [i - window_size, i] or [i - window_size[0], i].
            If causal mode is disabled, Query at position i will only attend to keys between [i - window_size, i + window_size] or [i - window_size[0], i + window_size[1]].
        return_softmax_lse (bool): Whether to return the log-sum-exp of the softmax. Default is False.
        return_seed_offset (bool): Whether to return the random seed offset. Default is False.
        fixed_seed_of fset(Tensor, optional): With fixed seed, offset for dropout mask.
        rng_name (str): The name to select Generator.
        training (bool): Whether the module is in training mode. Default is True.
        name (str, optional): Name of the operation. Default is None. Normally, users do not need to set this property.
            For more information, refer to :ref:`api_guide_Name` .

    Returns
        Tensor. The computed attention result with the same shape as the input `query`.

    Warning:
        This API only supports inputs with dtype float16 and bfloat16.

    Hint:
        This API supports GQA.

    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP('flash_attn need A100 compile')
            >>> import paddle
            >>> paddle.seed(2023)
            >>> q = paddle.rand((1, 10, 2, 32),dtype="bfloat16") # shape: [batch_size, seq_len, num_heads, head_dim]
            >>> k = paddle.rand((1, 10, 2, 32),dtype="bfloat16") # shape: [batch_size, seq_len, num_heads, head_dim]
            >>> v = paddle.rand((1, 10, 2, 32),dtype="bfloat16") # shape: [batch_size, seq_len, num_heads, head_dim]
            >>> startend_row_indices = paddle.to_tensor([8]*10 + [5]*10, dtype="int32").reshape([1, 2, 10, 1])
            >>> output = paddle.nn.functional.flashmask_attention(q, k, v, startend_row_indices, causal=True)
            >>> print(output)
            Tensor(shape=[1, 10, 2, 32], dtype=bfloat16, place=Place(gpu:0), stop_gradient=True,
                [[[[0.82421875, 0.27539062, 0.80859375, 0.98046875, 0.00251770,
                    0.41992188, 0.17285156, 0.11767578, 0.42773438, 0.31250000,
                    0.34570312, 0.70312500, 0.29296875, 0.44531250, 0.51562500,
                    0.96093750, 0.85546875, 0.15625000, 0.34765625, 0.98437500,
                    0.96484375, 0.45312500, 0.33593750, 0.56640625, 0.07714844,
                    0.43750000, 0.83984375, 0.66796875, 0.93750000, 0.24804688,
                    0.51171875, 0.55468750],
                    [0.54687500, 0.74609375, 0.43164062, 0.32421875, 0.10693359,
                    0.37304688, 0.53906250, 0.17187500, 0.57421875, 0.75000000,
                    0.13378906, 0.57031250, 0.19531250, 0.01403809, 0.29101562,
                    0.14257812, 0.07568359, 0.88671875, 0.75390625, 0.17089844,
                    0.87109375, 0.93359375, 0.89843750, 0.58203125, 0.75390625,
                    0.27539062, 0.67968750, 0.24804688, 0.57812500, 0.67578125,
                    0.92578125, 0.98046875]],

                    [[0.59765625, 0.62890625, 0.62109375, 0.75781250, 0.03295898,
                    0.64062500, 0.27929688, 0.20800781, 0.72265625, 0.52343750,
                    0.53125000, 0.61718750, 0.57421875, 0.56640625, 0.65625000,
                    0.48242188, 0.68359375, 0.42968750, 0.26562500, 0.86718750,
                    0.83203125, 0.40820312, 0.38281250, 0.59765625, 0.43945312,
                    0.22851562, 0.86328125, 0.51562500, 0.89453125, 0.62500000,
                    0.50390625, 0.67968750],
                    [0.34765625, 0.61328125, 0.58593750, 0.60156250, 0.43164062,
                    0.41601562, 0.71093750, 0.59765625, 0.53515625, 0.78125000,
                    0.13867188, 0.30664062, 0.48828125, 0.04394531, 0.24316406,
                    0.18847656, 0.10644531, 0.71093750, 0.69140625, 0.35937500,
                    0.44531250, 0.81640625, 0.44140625, 0.64062500, 0.81640625,
                    0.61328125, 0.72265625, 0.53125000, 0.49414062, 0.59765625,
                    0.54296875, 0.61328125]],

                    [[0.65234375, 0.47656250, 0.71875000, 0.64843750, 0.23828125,
                    0.61328125, 0.29101562, 0.26562500, 0.54296875, 0.60937500,
                    0.67187500, 0.67578125, 0.64062500, 0.41406250, 0.47656250,
                    0.40820312, 0.66406250, 0.39453125, 0.39453125, 0.62109375,
                    0.58593750, 0.31054688, 0.31835938, 0.45703125, 0.52343750,
                    0.43164062, 0.64453125, 0.49804688, 0.82812500, 0.48242188,
                    0.38476562, 0.59375000],
                    [0.44921875, 0.62109375, 0.50390625, 0.51562500, 0.51953125,
                    0.57812500, 0.78515625, 0.73437500, 0.60546875, 0.55078125,
                    0.30273438, 0.23339844, 0.60546875, 0.33007812, 0.23242188,
                    0.30468750, 0.34570312, 0.70703125, 0.72656250, 0.58593750,
                    0.40234375, 0.62109375, 0.62109375, 0.69531250, 0.66796875,
                    0.51562500, 0.45898438, 0.67968750, 0.48828125, 0.50000000,
                    0.54687500, 0.71875000]],

                    [[0.67578125, 0.50000000, 0.58203125, 0.62109375, 0.43554688,
                    0.69531250, 0.30273438, 0.24023438, 0.57812500, 0.63671875,
                    0.51171875, 0.52734375, 0.60546875, 0.45507812, 0.42382812,
                    0.46093750, 0.55859375, 0.34960938, 0.39453125, 0.57031250,
                    0.55078125, 0.47265625, 0.24609375, 0.51953125, 0.46093750,
                    0.49218750, 0.49609375, 0.60156250, 0.76953125, 0.57421875,
                    0.40429688, 0.57031250],
                    [0.45703125, 0.71093750, 0.58984375, 0.43164062, 0.54296875,
                    0.57031250, 0.72265625, 0.61328125, 0.64453125, 0.50781250,
                    0.28125000, 0.19531250, 0.60546875, 0.40625000, 0.18554688,
                    0.33203125, 0.40039062, 0.58593750, 0.79687500, 0.45507812,
                    0.32812500, 0.58203125, 0.70703125, 0.64453125, 0.53906250,
                    0.57421875, 0.48828125, 0.53515625, 0.49804688, 0.50000000,
                    0.48437500, 0.55468750]],

                    [[0.64453125, 0.43164062, 0.54687500, 0.53125000, 0.42187500,
                    0.71484375, 0.30273438, 0.21484375, 0.50390625, 0.69531250,
                    0.58203125, 0.51562500, 0.61328125, 0.41992188, 0.40039062,
                    0.46679688, 0.58984375, 0.39062500, 0.41992188, 0.49023438,
                    0.47851562, 0.47070312, 0.30078125, 0.50390625, 0.47656250,
                    0.44921875, 0.43164062, 0.63671875, 0.78125000, 0.60156250,
                    0.48242188, 0.58203125],
                    [0.52343750, 0.69921875, 0.58984375, 0.35156250, 0.49218750,
                    0.58593750, 0.71093750, 0.59375000, 0.66406250, 0.49414062,
                    0.24023438, 0.18554688, 0.66796875, 0.50000000, 0.23144531,
                    0.29882812, 0.49414062, 0.57031250, 0.70312500, 0.42773438,
                    0.35351562, 0.47460938, 0.73437500, 0.53125000, 0.47070312,
                    0.49609375, 0.50000000, 0.55078125, 0.50000000, 0.45898438,
                    0.45703125, 0.61328125]],

                    [[0.63671875, 0.41210938, 0.52734375, 0.56640625, 0.44531250,
                    0.64843750, 0.37890625, 0.31250000, 0.56640625, 0.62890625,
                    0.53125000, 0.51562500, 0.54296875, 0.50781250, 0.35546875,
                    0.41601562, 0.55468750, 0.36914062, 0.35937500, 0.45117188,
                    0.46875000, 0.49609375, 0.28710938, 0.50000000, 0.49609375,
                    0.50000000, 0.51562500, 0.57031250, 0.77734375, 0.62109375,
                    0.43164062, 0.50781250],
                    [0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        ]],

                    [[0.62109375, 0.44531250, 0.46875000, 0.61328125, 0.39062500,
                    0.60156250, 0.41015625, 0.28710938, 0.58984375, 0.67968750,
                    0.55859375, 0.48632812, 0.51562500, 0.42382812, 0.37695312,
                    0.46679688, 0.54687500, 0.44921875, 0.33789062, 0.36328125,
                    0.49023438, 0.44140625, 0.25000000, 0.45312500, 0.43945312,
                    0.45507812, 0.46679688, 0.57812500, 0.65625000, 0.64062500,
                    0.42382812, 0.57031250],
                    [0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        ]],

                    [[0.62500000, 0.47070312, 0.51562500, 0.61328125, 0.36718750,
                    0.66406250, 0.37890625, 0.28320312, 0.65625000, 0.66015625,
                    0.48632812, 0.53906250, 0.46679688, 0.47851562, 0.43359375,
                    0.45703125, 0.47070312, 0.39843750, 0.32617188, 0.37304688,
                    0.49023438, 0.50390625, 0.27148438, 0.46679688, 0.37695312,
                    0.49023438, 0.47265625, 0.58593750, 0.64453125, 0.60156250,
                    0.38476562, 0.62109375],
                    [0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        ]],

                    [[0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        ],
                    [0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        ]],

                    [[0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        ],
                    [0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        , 0.        , 0.        , 0.        ,
                    0.        , 0.        ]]]])
            >>> # doctest: -SKIP


    To convert FlashMask's `startend_row_indices` to `dense_mask`, use the code below:

    .. code-block:: python

        >>> import paddle
        >>> import numpy as np
        >>> def flashmask_to_densemask(startend_row_indices, dtype, causal=True):
        ...     if startend_row_indices is None:
        ...         return None
        ...     bz, num_head, seq_len, bound_num = startend_row_indices.shape
        ...     m = paddle.zeros((bz, num_head, seq_len, seq_len), dtype=dtype)
        ...     has_end = (causal and bound_num == 2) or ((not causal) and bound_num == 4)
        ...     for bi in range(bz):
        ...         for hi in range(num_head):
        ...             for j in range(seq_len):
        ...                 downstart = startend_row_indices[bi, hi, j, 0]
        ...                 if has_end:
        ...                     downend = startend_row_indices[bi, hi, j, 1]
        ...                     m[bi, hi, downstart:downend, j] = -np.inf
        ...                 else:
        ...                     m[bi, hi, downstart:, j] = -np.inf
        ...                 if causal:
        ...                     m[bi, hi, :j, j] = -np.inf
        ...                 else:
        ...                     if has_end:
        ...                         upstart = startend_row_indices[bi, hi, j, 2]
        ...                         upend = startend_row_indices[bi, hi, j, 3]
        ...                         m[bi, hi, upstart:upend, j] = -np.inf
        ...                     else:
        ...                         upend = startend_row_indices[bi, hi, j, 1]
        ...                         m[bi, hi, :upend, j] = -np.inf
        ...     return m

    For `Causal Mask`, where `causal=True`, the values of `startend_row_indices` are as follows:

    .. code-block:: python

       [[[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]])

        >>> # doctest: +SKIP('Only example')
        >>> import paddle
        >>> startend_row_indices = paddle.to_tensor([8]*10, dtype="int32").reshape([1, 1, 10, 1])
        >>> print(startend_row_indices)
        Tensor(shape=[1, 1, 10, 1], dtype=int32, place=Place(gpu:0), stop_gradient=True,
            [[[[8],
                [8],
                [8],
                [8],
                [8],
                [8],
                [8],
                [8],
                [8],
                [8]]]])
        >>> # doctest: -SKIP


    For `Sliding Window Mask`, where `causal=True`, the values of `startend_row_indices` are as follows:

    .. code-block:: python

       [[[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
          [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
          [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]]]])

        >>> # doctest: +SKIP('Only example')
        >>> import paddle
        >>> startend_row_indices = paddle.to_tensor([3, 4, 5, 6, 7, 8, 9, 10, 10, 10], dtype="int32").reshape([1, 1, 10, 1])
        >>> print(startend_row_indices)
        Tensor(shape=[1, 1, 10, 1], dtype=int32, place=Place(gpu:0), stop_gradient=True,
            [[[[3 ],
                [4 ],
                [5 ],
                [6 ],
                [7 ],
                [8 ],
                [9 ],
                [10],
                [10],
                [10]]]])
        >>> # doctest: -SKIP

    For `Causal Document Mask`, where `causal=True`, the values of `startend_row_indices` are as follows:

    .. code-block:: python

       [[[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
          [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]]]])

        >>> # doctest: +SKIP('Only example')
        >>> import paddle
        >>> startend_row_indices = paddle.to_tensor([4, 4, 4, 4, 7, 7, 7, 10, 10, 10], dtype="int32").reshape([1, 1, 10, 1])
        >>> print(startend_row_indices)
        Tensor(shape=[1, 1, 10, 1], dtype=int32, place=Place(gpu:0), stop_gradient=True,
            [[[[4 ],
                [4 ],
                [4 ],
                [4 ],
                [7 ],
                [7 ],
                [7 ],
                [10],
                [10],
                [10]]]])
        >>> # doctest: -SKIP

    For `Document Mask`, where `causal=False`, the values of `startend_row_indices` are as follows:

    .. code-block:: python

       [[[[1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
          [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
          [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
          [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
          [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]]]])

        >>> # doctest: +SKIP('Only example')
        >>> import paddle
        >>> LTS = paddle.to_tensor([4, 4, 4, 4, 7, 7, 7, 10, 10, 10], dtype="int32").reshape([1, 1, 10, 1])
        >>> UTE = paddle.to_tensor([0, 0, 0, 0, 4, 4, 4, 7, 7, 7], dtype="int32").reshape([1, 1, 10, 1])
        >>> startend_row_indices = paddle.concat([LTS, UTE], axis=-1)
        >>> print(startend_row_indices)
        Tensor(shape=[1, 1, 10, 2], dtype=int32, place=Place(gpu:0), stop_gradient=True,
            [[[[4 , 0 ],
                [4 , 0 ],
                [4 , 0 ],
                [4 , 0 ],
                [7 , 4 ],
                [7 , 4 ],
                [7 , 4 ],
                [10, 7 ],
                [10, 7 ],
                [10, 7 ]]]])
        >>> # doctest: -SKIP

    For `Share Question Mask`, where `causal=True`, the values of `startend_row_indices` are as follows:

    .. code-block:: python

       [[[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
          [1, 1, 1, 1, 0, 0, 0, 1, 0, 0],
          [1, 1, 1, 1, 0, 0, 0, 1, 1, 0],
          [1, 1, 1, 1, 0, 0, 0, 1, 1, 1]]]])

        >>> # doctest: +SKIP('Only example')
        >>> import paddle
        >>> startend_row_indices = paddle.to_tensor([10, 10, 10, 10, 7, 7, 7, 10, 10, 10], dtype="int32").reshape([1, 1, 10, 1])
        >>> print(startend_row_indices)
        Tensor(shape=[1, 1, 10, 1], dtype=int32, place=Place(gpu:0), stop_gradient=True,
            [[[[10],
                [10],
                [10],
                [10],
                [7 ],
                [7 ],
                [7 ],
                [10],
                [10],
                [10]]]])
        >>> # doctest: -SKIP

    For `Global + Sliding Window Mask`, where `causal=False`, the values of `startend_row_indices` are as follows:

    .. code-block:: python

        >>> # doctest: +SKIP('Only example')

       [[[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
          [1, 1, 0, 1, 1, 1, 0, 0, 0, 0],
          [1, 1, 0, 0, 1, 1, 1, 0, 0, 0],
          [1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
          [1, 1, 0, 0, 0, 0, 1, 1, 1, 0],
          [1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
          [1, 1, 0, 0, 0, 0, 0, 0, 1, 1]]]])

        >>> import paddle
        >>> LTS = paddle.to_tensor([10, 10, 4, 5, 6, 7, 8, 9, 10, 10], dtype="int32").reshape([1, 1, 10, 1])
        >>> LTE = paddle.to_tensor([10, 10, 10, 10, 10, 10, 10, 10, 10, 10], dtype="int32").reshape([1, 1, 10, 1])
        >>> UTS = paddle.to_tensor([0, 0, 0, 0, 2, 2, 2, 2, 2, 2], dtype="int32").reshape([1, 1, 10, 1])
        >>> UTE = paddle.to_tensor([0, 0, 0, 0, 3, 4, 5, 6, 7, 8], dtype="int32").reshape([1, 1, 10, 1])
        >>> startend_row_indices = paddle.concat([LTS, LTE, UTS, UTE], axis=-1)
        >>> print(startend_row_indices)
        Tensor(shape=[1, 1, 10, 4], dtype=int32, place=Place(gpu:0), stop_gradient=True,
            [[[[10, 10, 0 , 0 ],
                [10, 10, 0 , 0 ],
                [4 , 10, 0 , 0 ],
                [5 , 10, 0 , 0 ],
                [6 , 10, 2 , 3 ],
                [7 , 10, 2 , 4 ],
                [8 , 10, 2 , 5 ],
                [9 , 10, 2 , 6 ],
                [10, 10, 2 , 7 ],
                [10, 10, 2 , 8 ]]]])
        >>> # doctest: -SKIP

    For `Causal Blockwise Mask`, where `causal=True`, the values of `startend_row_indices` are as follows:

    .. code-block:: python

       [[[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]]])

        >>> # doctest: +SKIP('Only example')
        >>> import paddle
        >>> LTS = paddle.to_tensor([4, 4, 4, 4, 10, 10, 10, 10, 10, 10], dtype="int32").reshape([1, 1, 10, 1])
        >>> LTE = paddle.to_tensor([7, 7, 7, 7, 10, 10, 10, 10, 10, 10], dtype="int32").reshape([1, 1, 10, 1])
        >>> startend_row_indices = paddle.concat([LTS, LTE], axis=-1)
        >>> print(startend_row_indices)
        Tensor(shape=[1, 1, 10, 2], dtype=int32, place=Place(gpu:0), stop_gradient=True,
            [[[[4 , 7 ],
                [4 , 7 ],
                [4 , 7 ],
                [4 , 7 ],
                [10, 10],
                [10, 10],
                [10, 10],
                [10, 10],
                [10, 10],
                [10, 10]]]])
        >>> # doctest: -SKIP

    For `Prefix LM Document Mask`, where `causal=False`, the values of `startend_row_indices` are as follows:

    .. code-block:: python

       [[[[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
          [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
          [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]]])

        >>> # doctest: +SKIP('Only example')
        >>> import paddle
        >>> LTS = paddle.to_tensor([3, 3, 3, 5, 5, 10, 10, 10, 10, 10], dtype="int32").reshape([1, 1, 10, 1])
        >>> UTE = paddle.to_tensor([0, 0, 2, 3, 3, 5, 5, 7, 8, 9], dtype="int32").reshape([1, 1, 10, 1])
        >>> startend_row_indices = paddle.concat([LTS, UTE], axis=-1)
        >>> print(startend_row_indices)
        Tensor(shape=[1, 1, 10, 2], dtype=int32, place=Place(gpu:0), stop_gradient=True,
            [[[[3 , 0 ],
                [3 , 0 ],
                [3 , 2 ],
                [5 , 3 ],
                [5 , 3 ],
                [10, 5 ],
                [10, 5 ],
                [10, 7 ],
                [10, 8 ],
                [10, 9 ]]]])
        >>> # doctest: -SKIP

    For `Prefix LM Causal Mask`, where `causal=False`, the values of `startend_row_indices` are as follows:

    .. code-block:: python

       [[[[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]]])

        >>> # doctest: +SKIP('Only example')
        >>> import paddle
        >>> LTS = paddle.to_tensor([10, 10, 10, 10, 10, 10, 10, 10, 10, 10], dtype="int32").reshape([1, 1, 10, 1])
        >>> UTE = paddle.to_tensor([0, 0, 0, 0, 0, 5, 6, 7, 8, 9], dtype="int32").reshape([1, 1, 10, 1])
        >>> startend_row_indices = paddle.concat([LTS, UTE], axis=-1)
        >>> print(startend_row_indices)
        Tensor(shape=[1, 1, 10, 2], dtype=int32, place=Place(gpu:0), stop_gradient=True,
            [[[[10, 0 ],
                [10, 0 ],
                [10, 0 ],
                [10, 0 ],
                [10, 0 ],
                [10, 5 ],
                [10, 6 ],
                [10, 7 ],
                [10, 8 ],
                [10, 9 ]]]])

    For `QK-sparse Mask`, where `causal=True`, the values of `startend_row_indices` are as follows:

    .. code-block:: python

       [[[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]]])

        >>> # doctest: +SKIP('Only example')
        >>> import paddle
        >>> LTS = paddle.to_tensor([10, 10, 2, 3, 4, 5, 6, 7, 10, 10], dtype="int32").reshape([1, 1, 10, 1])
        >>> LTE = paddle.to_tensor([10, 10, 5, 5, 5, 5, 8, 8, 10, 10], dtype="int32").reshape([1, 1, 10, 1])
        >>> startend_row_indices = paddle.concat([LTS, LTE], axis=-1)
        >>> print(startend_row_indices)
        Tensor(shape=[1, 1, 10, 2], dtype=int32, place=Place(gpu:0), stop_gradient=True,
            [[[[10, 10],
                [10, 10],
                [2 , 5 ],
                [3 , 5 ],
                [4 , 5 ],
                [5 , 5 ],
                [6 , 8 ],
                [7 , 8 ],
                [10, 10],
                [10, 10]]]])

        >>> # doctest: -SKIP
    """
    if window_size is not None:
        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        sq = query.shape[1]
        bsz = query.shape[0]
        assert (
            startend_row_indices is None
        ), "can't use window_size with startend_row_indices"
        if causal:
            startend_row_indices = paddle.arange(
                window_size[0] + 1, sq + window_size[0] + 1, dtype="int32"
            ).reshape((1, 1, sq, 1))
            startend_row_indices = paddle.clip(
                startend_row_indices, max=sq
            ).repeat_interleave(bsz, 0)

        else:
            startend_row_indices = paddle.empty((1, 1, sq, 2), dtype="int32")
            startend_row_indices[0, 0, :, 0] = paddle.arange(
                window_size[0] + 1, sq + window_size[0] + 1, dtype="int32"
            )
            startend_row_indices[0, 0, :, 1] = paddle.arange(
                -window_size[1], sq - window_size[1], dtype="int32"
            )
            startend_row_indices = paddle.clip(
                startend_row_indices, min=0, max=sq
            ).repeat_interleave(bsz, 0)

    if startend_row_indices is None:
        (
            out,
            result_softmax,
            result_softmax_lse,
            result_seed_offset,
        ) = _C_ops.flash_attn(
            query,
            key,
            value,
            fixed_seed_offset,
            None,
            dropout,
            causal,
            False,
            not training,
            rng_name,
        )

    else:
        assert (
            startend_row_indices.dtype == paddle.int32
        ), f"startend_row_indices.dtype must be paddle.int32, but got {startend_row_indices.dtype}"
        assert (
            len(startend_row_indices.shape) == 4
        ), f"startend_row_indices rank must be 4,but got {startend_row_indices.shape}"

        assert (
            startend_row_indices.shape[0] == key.shape[0]
        ), f"startend_row_indices.shape[0] must be equal to batch_size, but got {startend_row_indices.shape[0]} and {key.shape[0]}"

        assert (
            startend_row_indices.shape[2] == key.shape[1]
        ), f"startend_row_indices.shape[2] must be equal to seqlen_k, but got {startend_row_indices.shape[2]} and {key.shape[2]}"
        assert startend_row_indices.shape[1] in [
            1,
            key.shape[2],
        ], "startend_row_indices head_num must be equal to 1(broadcast) or hean_num_k."

        if causal:
            if startend_row_indices.shape[-1] == 1:
                has_end = False
            elif startend_row_indices.shape[-1] == 2:
                has_end = True
            else:
                raise ValueError(
                    f"Invalid shape of startend_row_indices, when causal is True, the last dimension should be either 1 or 2 but got {startend_row_indices.shape[-1]}"
                )
        else:
            if startend_row_indices.shape[-1] == 2:
                has_end = False
            elif startend_row_indices.shape[-1] == 4:
                has_end = True
            else:
                raise ValueError(
                    f"Invalid shape of startend_row_indices, when causal is False, the last dimension should be either 2 or 4 but got {startend_row_indices.shape[-1]}"
                )

        (
            out,
            result_softmax,
            result_softmax_lse,
            result_seed_offset,
        ) = _C_ops.flashmask_attention(
            query,
            key,
            value,
            startend_row_indices,
            fixed_seed_offset,
            dropout,
            causal,
            False,
            not training,
            rng_name,
        )

    outputs = [out]
    if return_softmax_lse:
        outputs += [result_softmax_lse]
    if return_seed_offset:
        outputs += [result_seed_offset]
    if len(outputs) == 1:
        return outputs[0]
    else:
        return outputs


def calc_reduced_attention_scores(
    query: paddle.Tensor, key: paddle.Tensor, softmax_lse: paddle.Tensor
) -> paddle.Tensor:
    r"""
    The equation is:

    .. math::

        result=reduce\_sum(softmax(\frac{ Q * K^T }{\sqrt{d}}), dim=-2)

    Warning:
        This API only supports inputs with dtype float16 and bfloat16.

    Args:
        query(Tensor): The query tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seqlen_q, num_heads, head_dim].
                        The dtype can be float16 or bfloat16.
        key(Tensor): The key tensor in the Attention module.
                        4-D tensor with shape:
                        [batch_size, seqlen_k, num_heads, head_dim].
                        The dtype can be float16 or bfloat16.
        softmax_lse(Tensor): The logsumexp of each row returned by _C_ops.flash_attn().
                        3-D tensor with shape:
                        [batch_size, num_heads, seqlen_q_rounded], where seqlen_q_rounded = ceil(seqlen_q/128).
                        The dtype is float32.
    Returns:
        reduced_attention_scores(Tensor), The reduce sum of attention scores across seqlen_q.
        4-D tensor with shape: [batch_size, num_heads, 1, seqlen_k]. The dtype is float32.
    Examples:
        .. code-block:: python

            >>> # doctest: +SKIP('reduce_attn_scores need A100 compile')
            >>> import paddle
            >>> import numpy as np
            >>> import paddle._C_ops as _C_ops
            >>> from paddle.nn.functional.flash_attention import (
            >>>     calc_reduced_attention_scores
            >>> )
            >>> np.random.seed(2024)
            >>> q_shape = (5,1024,16,128)
            >>> k_shape = (5,2048,16,128)
            >>> dtype = 'float16'
            >>> query = np.random.random(q_shape)
            >>> key = np.random.random(k_shape)
            >>> q = paddle.to_tensor(
            >>>     query, place=place, dtype=dtype, stop_gradient=True
            >>> )
            >>> k = paddle.to_tensor(
            >>>     key, place=place, dtype=dtype, stop_gradient=True
            >>> )
            >>> _, _, softmax_lse, _ = _C_ops.flash_attn(
            >>>     q,
            >>>     k,
            >>>     k,
            >>>     (None,), #fixed_seed_offset
            >>>     None, #attn_mask
            >>>     0.0, #dropout
            >>>     False, #causal
            >>>     False, #return_softmax
            >>>     False, #is_test
            >>>     "" #rng_name
            >>> )
            >>> reduced_attn_scores = calc_reduced_attention_scores(
            >>>     q,
            >>>     k,
            >>>     softmax_lse,
            >>> )
            >>> # doctest: -SKIP
    """
    assert (
        query.stop_gradient and key.stop_gradient
    ), 'calc_reduced_attention_scores() is for inference only.'

    if in_dynamic_or_pir_mode():
        reduced_scores = _C_ops.calc_reduced_attn_scores(
            query, key, softmax_lse
        )
        return reduced_scores

    helper = LayerHelper('calc_reduced_attn_scores', **locals())
    reduced_scores = helper.create_variable_for_type_inference(paddle.float32)
    softmax = helper.create_variable_for_type_inference(paddle.float32)
    inputs = {
        'q': query,
        'k': key,
        'softmax_lse': softmax_lse,
    }
    outputs = {
        'reduced_scores': reduced_scores,
    }
    helper.append_op(
        type='calc_reduced_attn_scores',
        inputs=inputs,
        outputs=outputs,
    )
    return reduced_scores
