
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import paddle
import triton
import triton.language as tl

paddle.seed(123)

@triton.jit
def conv_kernel(
    activation_ptr,  
    weight_ptr,
    output_ptr,
    batch, ic, ih, iw,
    oh, ow, oc,
    KH: tl.constexpr, KW: tl.constexpr,
    dilation_h: tl.constexpr, dilation_w: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    ic = tl.multiple_of(ic, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(oh * ow * batch, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(oc, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    
    offs_batch = offs_m  // (oh * ow)
    offs_oh = offs_m % (oh * ow) // ow
    offs_ow = offs_m % (oh * ow) % ow

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    weight_ptrs = weight_ptr + offs_k[:,None] + offs_n[None,:] * KH * KW * ic
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for kh in range(0, KH):
        for kw in range(0, KW):
            offs_ih = offs_oh * STRIDE_H + kh * dilation_h - PAD_H
            # offs_ih = tl.where(offs_ih < 0, 0, offs_ih)
            # offs_ih = tl.where(offs_ih < ih, offs_ih, ih - 1)
            
            offs_iw = offs_ow * STRIDE_W + kw * dilation_w - PAD_W
            # offs_iw = tl.where(offs_iw < 0, 0, offs_iw)
            # offs_iw = tl.where(offs_iw < iw, offs_iw, iw - 1)
            mask = offs_ih[:, None] < ih and offs_ih[:, None] >= 0
            mask = mask and offs_iw[:, None] < iw
            mask = mask and offs_iw[:, None] >= 0
            
            activation_ptrs = activation_ptr + offs_batch[:,None] * ih * iw * ic + offs_ih[:,None] * iw * ic + offs_iw[:,None] * ic + offs_k[None, :]
            
            for k in range(0, tl.cdiv(ic, BLOCK_SIZE_K)):

                # mask_k_1 = offs_k[None, :] < ic - k * BLOCK_SIZE_K
                # mask1 = mask and mask_k_1
                # mask_k_2 = offs_k[:, None] < ic - k * BLOCK_SIZE_K

                activation = tl.load(activation_ptrs, mask = mask, other=0.0) 
                weight = tl.load(weight_ptrs)
                
                accumulator += tl.dot(activation, weight)
                
                weight_ptrs += BLOCK_SIZE_K * 1
                activation_ptrs += BLOCK_SIZE_K * 1
    
    output_ptrs = output_ptr + offs_m[:, None] * oc + offs_n[None, :]
    tl.store(output_ptrs, accumulator)

def conv(activation_tensor, weight_tensor):
    batch, ih, iw, ic = activation_tensor.shape
    oc = weight_tensor.shape[0]
    stride_h = 2
    stride_w = 2
    pad_h0 = 1
    pad_h1 = 1
    pad_w0 = 1
    pad_w1 = 1
    dilation_h = 1
    dilation_w = 1
    kh = 3
    kw = 3
    oh = (ih + pad_h0 + pad_h1 - dilation_h * (kh - 1) - 1) // stride_h + 1
    ow = (iw + pad_w0 + pad_w1 - dilation_w * (kw - 1) - 1) // stride_w + 1
    output = paddle.zeros([batch, oh, ow, oc])
    M = batch * oh * ow
    N = oc
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    # for i in range(100):
    conv_kernel[grid](
        activation_tensor, weight_tensor, output,
        batch, ic, ih, iw,
        oh, ow, oc,
        # 下面是超参
        KH = kh, KW = kw,
        STRIDE_H = stride_h, STRIDE_W = stride_w,
        PAD_H = pad_h0, PAD_W = pad_w0,
        dilation_h = dilation_h, dilation_w = dilation_w,
        BLOCK_SIZE_M = 128,
        BLOCK_SIZE_N = 128,
        BLOCK_SIZE_K = 16,
    )
    return output

# 前提条件 ic%BLOCK_SIZE_K == 0  
batch = 16
ic = 256
ih = 64
iw = 64
oc = 512
activation_size = (batch, ih, iw, ic)
weight_size = (oc, 3, 3, ic)

x = paddle.rand(activation_size)
#print(x)
y = paddle.rand(weight_size)-0.5
output_triton = conv(x, y)

#print(output_triton)

import paddle
import paddle.nn as nn

paddle.disable_static()
weight_attr = paddle.ParamAttr(name="weight",
                               initializer = paddle.nn.initializer.Assign(y.transpose([0,3,1,2]).numpy()),
                               learning_rate=0.5,
                               regularizer=paddle.regularizer.L2Decay(1.0),
                               trainable=False)
conv_ = nn.Conv2D(ic, oc, (3, 3), stride = (2,2), padding=1, dilation=1, weight_attr=weight_attr, data_format='NCHW', padding_mode='zeros')


import datetime
repeat_times = 100
warm_up_times = 20
# paddle test the time
for i in range(warm_up_times):
    y_var = conv_(x.transpose([0, 3, 1, 2]))
paddle.device.cuda.synchronize(0)
starttime = datetime.datetime.now()
for i in range(repeat_times):
    y_var = conv_(x.transpose([0, 3, 1, 2]))
paddle.device.cuda.synchronize(0) 
endtime = datetime.datetime.now()
duringtime = endtime - starttime
time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
print(f"paddle {time_ms} ms")
# diff = y_var.transpose([0, 2, 3, 1]) - output_triton
# print(paddle.max(diff[0]))
# print(paddle.max(diff[1]))
# triton test the time
for i in range(warm_up_times):
    output_triton = conv(x, y)
paddle.device.cuda.synchronize(0) 
starttime = datetime.datetime.now()
for i in range(repeat_times):
    output_triton = conv(x, y)
paddle.device.cuda.synchronize(0)
endtime = datetime.datetime.now()
duringtime = endtime - starttime
time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
print(f"triton {time_ms} ms")
print(paddle.max(y_var.transpose([0, 2, 3, 1]) - output_triton))

