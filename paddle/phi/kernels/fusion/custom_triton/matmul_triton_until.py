import paddle
import triton
import triton.language as tl

# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `_matmul`.
@triton.jit
def leaky_relu(x):
    x = x + 1
    return tl.where(x >= 0, x, 0.01 * x)

def relu(x):
    return tl.where(x >= 0, x, 0)