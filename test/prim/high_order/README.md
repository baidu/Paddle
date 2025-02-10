### paddle高阶微分单测使用说明
> 使用简单的语法来实现paddle函数的高阶微分测试

#### 如何写一个函数的高阶微分单测
以`paddle.sin`为例。
```
import sys
import unittest

import numpy as np

sys.path.append("../../legacy_test")

import autodiff_checker_helper as ad_checker
import parameterized as param

import paddle
from paddle.base import core

class TestSinHigherOrderAD(unittest.TestCase):
    # 参数化测试
    @param.parameterized.expand(
        [
            (paddle.sin, [2, 3, 2], 'float32', 4, ('cpu', 'gpu')),
            (paddle.sin, [2, 3, 2, 4], 'float64', 4, ('cpu', 'gpu')),
        ]
    )
    def test_high_order_autodiff(self, func, shape, dtype, order, places):
        var_1 = np.random.randn(*shape).astype(dtype)
        for place in places:
            if place == 'gpu' and not core.is_compiled_with_cuda():
                continue
            var1 = paddle.to_tensor(var_1, place=place)
            ad_checker.check_vjp(func, [var1], argnums=(0), order=order, atol=1e-3, rtol=1e-3)
```
主要分成三步：
1. 准备测试用例，主要有测试函数, 输入shape, 输入dtype, 计算的设备, 以及需要测试的阶数
2. 根据测试用例生成测试函数的输入
3. 调用autodiff_checker_helper下的check_vjp函数测试高阶微分是否正确


#### check_vjp使用说明
check_vjp函数签名如下：
```
def check_vjp(
    func, # 需要高阶测试的函数，是一个python func（需要是有paddle api的函数）
    args, # func 的输入，是一个tensor的列表，可以是list也可以是tuple，（不需要计算梯度的功能由argnums实现，args需要传入所有的输入tensor）
    kwargs={}, # func 的属性（除args以外的其他参数，以字典的形式传入）
    argnums=None, # list或者tuple，选择args中需要测试梯度的输入，默认None，会对对args所有输入测试
    order=2, # 测试的阶数，最低为2阶，如果设置为4，则会测试2，3，4阶
    atol=None, # 测试的绝对误差，默认情况下使用预先设置的数值
    rtol=None, # 测试的相对误差，默认情况下使用预先设置的数值
    eps=EPS, # 计算数值微分使用的扰动步长
):
```
> check_vjp 会依次计算测试函数的数值微分，解析微分（动态图+静态图），然后对比二者是否满足精度要求

[autodiff_checker_helper实现代码](https://github.com/PaddlePaddle/Paddle/blob/develop/test/legacy_test/autodiff_checker_helper.py)
