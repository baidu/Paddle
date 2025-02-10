import paddle
paddle.set_device('gpu:0')
a = paddle.load("./my.pdparams", return_numpy=False)

weights = a["weights"]
indices = a["indices"]
splines = a["splines"]

test = indices[0][0][splines[:,0, 1] < 0.15]
print("test = ", test, flush=True)
weights[0, test] = 0.2
