# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.fluid as fluid
from paddle.autograd.functional import vjp, Jacobian
from paddle.fluid.framework import in_dygraph_mode
import numpy as np


def _value_and_gradient(f, x, v=None):
    if in_dygraph_mode():
        value, gradient = vjp(f, paddle.to_tensor(x), v=v)
    else:
        JJ = paddle.autograd.functional.Jacobian(f, x)
        gradient = JJ[:][0]
        value = f(x)
    return value, gradient
