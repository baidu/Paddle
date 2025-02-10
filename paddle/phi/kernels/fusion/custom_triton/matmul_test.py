import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from triton_ops import triton_matmul

import paddle

a = paddle.rand([1024,1024],dtype="float16")
b = paddle.rand([1024,1024],dtype="float16")
bias = paddle.rand([1024],dtype="float16")
c1 = triton_matmul(a,b, None, False, False)
c2 = paddle.matmul(a,b)
print(paddle.max(c1-c2))
# exit(0)
import datetime
for i in range(10):
    c1 = triton_matmul(a,b,None, False, False)
paddle.device.cuda.synchronize(0)
starttime = datetime.datetime.now()
for i in range(100):
    c1 = triton_matmul(a,b, None, False, False)
    # c2 = paddle.matmul(a,b)
paddle.device.cuda.synchronize(0)
endtime = datetime.datetime.now()
duringtime = endtime - starttime
time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
print("The whoel end to end time : ", time_ms, "ms")

for i in range(10):
    c2 = paddle.matmul(a,b)
paddle.device.cuda.synchronize(0)
starttime = datetime.datetime.now()
for i in range(100):
    # c1 = triton_matmul(a,b)
    c2 = paddle.matmul(a,b)
paddle.device.cuda.synchronize(0)
endtime = datetime.datetime.now()
duringtime = endtime - starttime
time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
print("The whoel end to end time : ", time_ms, "ms")

