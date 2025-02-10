# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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


import os

os.environ['FLAGS_deny_cinn_ops'] = "transpose"
import time

import paddle

build_strategy = paddle.static.BuildStrategy()
build_strategy.build_cinn_pass = True

input_specs = [
    paddle.static.InputSpec(shape=[16, 1024, 1024], dtype='float32', name='var')
]


x = paddle.randn([16, 1204, 1024], dtype="float32")
x.stop_gradient = True


def test_fn_1(x):
    t1 = paddle.cos(x)
    t2 = paddle.transpose(t1, [1, 0, 2])
    t3 = paddle.sin(t2)
    t4 = paddle.transpose(t3, [1, 0, 2])
    t5 = paddle.cos(t4)
    t6 = paddle.nn.functional.sigmoid(t4)

    return t5, t6


compile_f_1 = paddle.jit.to_static(
    test_fn_1,
    build_strategy=build_strategy,
    input_spec=input_specs,
    full_graph=True,
)

out = compile_f_1(x)


st = time.time()

for i in range(10000):
    out = compile_f_1(x)

print("compile cost 1", time.time() - st)


def test_fn_2(x):
    t1 = paddle.cos(x)
    t3 = paddle.sin(t1)
    t5 = paddle.cos(t3)
    t6 = paddle.nn.functional.sigmoid(t3)

    return t5, t6


compile_f_2 = paddle.jit.to_static(
    test_fn_2,
    build_strategy=build_strategy,
    input_spec=input_specs,
    full_graph=True,
)


out2 = compile_f_2(x)

st = time.time()

for i in range(10000):
    out2 = compile_f_2(x)

print("compile cost 2", time.time() - st)
