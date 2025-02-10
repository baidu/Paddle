import paddle
from triton_ops import triton_FcRelu
from triton_ops import triton_Fc
from triton_ops import triton_Fc2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def test():
    B = 2
    M = 102400
    N = 768
    K = 4
    # paddle.set_device('gpu:0')
    a = paddle.rand((B, M, K),dtype ="float16")
    b = paddle.rand((K, N),dtype ="float16")
    bias = paddle.rand((N, ),dtype ="float16")
    triton_c = triton_FcRelu(a, b, bias)
    print(triton_c.shape)

    c = paddle.matmul(a, b)
    c = c + bias
    m = paddle.nn.ReLU()
    out = m(c)

    print(paddle.max(c - triton_c))

def test_benchMark(B, M, K, N):
    a = paddle.rand((B * M, K),dtype ="float16")
    b = paddle.rand((K, N),dtype ="float16")
    bias = paddle.rand((N, ),dtype ="float16")  
    import datetime
    repeat_times = 1
    warm_up_times = 0

    # triton op
    for i in range(warm_up_times):
        triton_c_1 = triton_FcRelu(a,b,bias)
    paddle.device.cuda.synchronize(0)   
    starttime = datetime.datetime.now()
    for i in range(repeat_times):
        triton_c_1 = triton_FcRelu(a,b,bias)
    paddle.device.cuda.synchronize(0)
    endtime = datetime.datetime.now()
    duringtime = endtime - starttime
    time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
    # print("The TRITON OP end to end time : ", time_ms, "ms")
    # print(paddle.max(c_1 - triton_c_1))
    re2 = time_ms
    
    # paddle op
    for i in range(warm_up_times):
        c_1 = paddle.matmul(a, b)
        c_1 = c_1 + bias
        m = paddle.nn.ReLU()
        c_1 = m(c_1)
    paddle.device.cuda.synchronize(0)

    starttime = datetime.datetime.now()
    for i in range(repeat_times):
        c_1 = paddle.matmul(a, b)
        c_1 = c_1 + bias
        m = paddle.nn.ReLU()
        c_1 = m(c_1)
    paddle.device.cuda.synchronize(0)
    endtime = datetime.datetime.now()
    duringtime = endtime - starttime
    time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
    # print("The PADDLE OP end to end time : ", time_ms, "ms")
    re1 = time_ms

    chazhi = paddle.max(c_1 - triton_c_1)
    # chazhi = a
    return re1, re2, chazhi


# if __name__ == "__main__":

#     B_SET = [1]
#     M_SET = [32, 64, 128, 1024]
#     K_SET = [64, 256, 2048, 4096]
#     N_SET = [32, 128, 2048, 4096]
#     print("|  M | K | N | matmul + relu | FcRelu_triton |")
#     print("|-----|-----|-----|-------|-------|")
#     # B = 16
#     # M = 128
#     # K = 256
#     # N = 128
#     # re1, re2, chazhi = test_benchMark(B, M, K, N)
#     # if(chazhi.numpy() > 0.0005):
#     #     exit(0)
#     # print(f"| {B} | {M} | {K} | {N} | {re1} | {re2}| ")
#     for B in B_SET:
#         for M in M_SET:
#             for K in K_SET:
#                 for N in K_SET:
#                     re1, re2, chazhi = test_benchMark(B, M, K, N)
#                     if(chazhi.numpy() > 0.0005):
#                         print(chazhi)
#                     print(f"| {B * M} | {K} | {N} | {re1} | {re2}| ")
# 16 | 128 | 2048 | 1024
# B = 5120

M = 256
K = 256
N = 256

a = paddle.rand((M, K),dtype ="float16")
b = paddle.rand((K, N),dtype ="float16")
c = paddle.transpose(b, perm=[1, 0])
# bias = paddle.rand((N, ),dtype ="float16") 
re1 = paddle.matmul(a, b)
re2 = triton_Fc(a, b)
re3 = triton_Fc2(a, c) 
print(paddle.max(re1 - re2))
print(paddle.max(re1 - re3))
import datetime
import time
repeat_times = 100
warm_up_times = 20

# paddle op
for i in range(warm_up_times):
    c_1 = paddle.matmul(a, b)
paddle.device.cuda.synchronize(0)

# starttime = datetime.datetime.now()
start_time = time.perf_counter()
for i in range(repeat_times):
    c_1 = paddle.matmul(a, b)
paddle.device.cuda.synchronize(0)
# endtime = datetime.datetime.now()
end_time = time.perf_counter()
#duringtime = endtime - starttime
#time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
time_ms = (end_time - start_time) * 1000
print("The PADDLE OP end to end time : ", time_ms, "ms")
re1 = time_ms

# triton op
for i in range(warm_up_times):
    triton_c_1 = triton_Fc(a,b)
paddle.device.cuda.synchronize(0)   
# starttime = datetime.datetime.now()
start_time = time.perf_counter()
for i in range(repeat_times):
    triton_c_1 = triton_Fc(a,b)
paddle.device.cuda.synchronize(0)
# endtime = datetime.datetime.now()
end_time = time.perf_counter()
# duringtime = endtime - starttime
# time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
time_ms = (end_time - start_time) * 1000
print("The TRITON OP end to end time : ", time_ms, "ms")
# print(paddle.max(c_1 - triton_c_1))
re2 = time_ms
# print((c_1 - triton_c_1)[0])
# print((c_1 - triton_c_1)[1])
# print((c_1 - triton_c_1)[2])
# print((c_1 - triton_c_1)[3])
# print((c_1 - triton_c_1)[4])
# print((c_1 - triton_c_1)[5])
print(paddle.max(c_1 - triton_c_1))



