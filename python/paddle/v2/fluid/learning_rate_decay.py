# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

import layers
from framework import Variable

__all__ = ['exponential_decay', ]


def exponential_decay(learning_rate,
                      global_step,
                      decay_steps,
                      decay_rate,
                      staircase=False):
    """
    ```python
    decayed_learning_rate = learning_rate *
            decay_rate ^ (global_step / decay_steps)
    ```
    """
    if global_step is None or not isinstance(global_step, Variable):
        raise ValueError("global_step is required for exponential_decay.")

    lr = learning_rate
    if not isinstance(lr, Variable):
        lr = layers.create_global_var(
            shape=[1], value=learning_rate, dtype='float32')

    decay_steps_var = layers.fill_constant(
        shape=[1], dtype='float32', value=decay_steps)
    decay_rate_var = layers.fill_constant(
        shape=[1], dtype='float32', value=decay_rate)

    # update learning_rate
    div_res = layers.elementwise_div(x=global_step, y=decay_steps_var)
    if staircase:
        div_res = layers.floor(x=div_res)
    pow_res = layers.elementwise_pow(x=decay_rate_var, y=div_res)
    return lr * pow_res
