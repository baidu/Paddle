import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from triton_ops import triton_fmha
from triton_ops import triton_fmha2
from triton_ops import triton_fmha3
import numpy as np
import paddle
from paddle.incubate.nn.functional import (
    variable_length_memory_efficient_attention,)
import os
import paddle.nn.functional as F

# 16， 32，512，128
batch = 1
seq_len = 512
heads = 32
head_dim = 128
q = paddle.rand((batch, heads, seq_len, head_dim),dtype ="float16")
k = paddle.rand((batch, heads, seq_len, head_dim),dtype ="float16")
v = paddle.rand((batch, heads, seq_len, head_dim),dtype ="float16")

# q = paddle.load('/nishirong/PaddleNLP/llm/q_out')
# k = paddle.load('/nishirong/PaddleNLP/llm/k_out')
# v = paddle.load('/nishirong/PaddleNLP/llm/v_out')


# seq_len = q.shape[2]
# print(q.shape)

seq_lens = paddle.to_tensor([[seq_len] * batch]).astype("int32")
scale=float(head_dim**-0.5)


# attn_mask1 = paddle.full([batch, 1, seq_len, seq_len], -10000.0, 'float16')
# for i in range(seq_len):
#     for j in range(0, i + 1):
#         attn_mask1[:,:,i,j] = 0.0

# out = paddle.matmul(q, k.transpose([0, 1, 3, 2]))
# out = out / (np.sqrt(head_dim))
# for k_m in range(seq_len):
#     for j in range(k_m+1, seq_len):
#         out[:,:,k_m,j] = -1000.0
# out = paddle.nn.functional.softmax(out, -1)
# out =  paddle.matmul(out, v)

# 正确性验证
qkv_out0_1 = variable_length_memory_efficient_attention(q, k, v, seq_lens, seq_lens, mask=None, scale=scale,causal=True)
qkv_out1_1 = triton_fmha(q, k, v, scale)
qkv_out2_1 = triton_fmha2(q, k, v, scale)
qkv_out2_1_ = triton_fmha2(q, k, v, scale)
qkv_out3_1 = triton_fmha3(q, k, v, scale)


print(paddle.max(qkv_out1_1 - qkv_out0_1))
# print((qkv_out2_1 - qkv_out0_1)[0][0][0])
# print((qkv_out2_1 - qkv_out0_1)[0][0][1])
print(paddle.max((qkv_out2_1 - qkv_out0_1)))
print(paddle.max(qkv_out3_1 - qkv_out0_1))

# for i in range(s):
#     print("================================")
#     print((qkv_out2_1 - qkv_out0_1)[0][0][i])
#     if(paddle.max((qkv_out2_1 - qkv_out0_1)[0][0][i]) > 0.05):
#         print(q[0][0][i])

#耗时验证
import datetime
repeat_times = 100
warm_up_times = 20

# paddle op
for i in range(warm_up_times):
    qkv_out0 = variable_length_memory_efficient_attention(q, k, v, seq_lens, seq_lens, mask=None, scale=scale, causal=True)  
paddle.device.cuda.synchronize(0)
starttime = datetime.datetime.now()
for i in range(repeat_times):
    qkv_out0 = variable_length_memory_efficient_attention(q, k, v, seq_lens, seq_lens, mask=None, scale=scale, causal=True)
paddle.device.cuda.synchronize(0)
endtime = datetime.datetime.now()
duringtime = endtime - starttime
time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
print("The PADDLE OP end to end time : ", time_ms, "ms")

# triton op1
for i in range(warm_up_times):
    qkv_out1 = triton_fmha(q , k, v, scale)
paddle.device.cuda.synchronize(0)
starttime = datetime.datetime.now()
for i in range(repeat_times):
    qkv_out1 = triton_fmha(q , k, v, scale)
paddle.device.cuda.synchronize(0)
endtime = datetime.datetime.now()
duringtime = endtime - starttime
time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
print("The TRITON OP 1 end to end time : ", time_ms, "ms")


# triton op2
for i in range(warm_up_times):
    qkv_out2 = triton_fmha2(q , k, v, scale)
paddle.device.cuda.synchronize(0)
starttime = datetime.datetime.now()
for i in range(repeat_times):
    qkv_out2 = triton_fmha2(q , k, v, scale)
paddle.device.cuda.synchronize(0)
endtime = datetime.datetime.now()
duringtime = endtime - starttime
time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
print("The TRITON OP 2 end to end time : ", time_ms, "ms")

# triton op3
for i in range(warm_up_times):
    qkv_out3 = triton_fmha3(q , k, v, scale)
paddle.device.cuda.synchronize(0)
starttime = datetime.datetime.now()
for i in range(repeat_times):
    qkv_out3 = triton_fmha3(q , k, v, scale)
paddle.device.cuda.synchronize(0)
endtime = datetime.datetime.now()
duringtime = endtime - starttime
time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
print("The TRITON OP 3 end to end time : ", time_ms, "ms")


print(paddle.max(qkv_out3 - qkv_out0))

print(paddle.max(qkv_out2 - qkv_out0))

print(paddle.max(qkv_out1 - qkv_out0))