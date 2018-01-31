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

__all__ = ['exponential_decay', 'natural_exp_decay', 'inverse_time_decay']
"""
When training a model, it's often useful to decay the
learning rate during training process, this is called
learning_rate_decay. There are many strategies to do
this, this module will provide some classical method.
User can also implement their own learning_rate_decay
strategy according to this module.
"""


def exponential_decay(learning_rate,
                      global_step,
                      decay_steps,
                      decay_rate,
                      staircase=False):
    """Applies exponential decay to the learning rate.

    ```python
    decayed_learning_rate = learning_rate *
            decay_rate ^ (global_step / decay_steps)
    ```
    Args:
        learning_rate: A scalar float32 value or a Variable. This
          will be the initial learning rate during training
        global_step: A Variable that record the training step.
        decay_steps: A Python `int32` number.
        decay_rate: A Python `float` number.
        staircase: Boolean. If set true, decay the learning rate every decay_steps.

    Returns:
        The decayed learning rate
    """
    if not isinstance(global_step, Variable):
        raise ValueError("global_step is required for exponential_decay.")

    # update learning_rate
    div_res = global_step / decay_steps
    if staircase:
        div_res = layers.floor(x=div_res)
    return learning_rate * (decay_rate**div_res)


def natural_exp_decay(learning_rate,
                      global_step,
                      decay_steps,
                      decay_rate,
                      staircase=False):
    """Applies natural exponential decay to the initial learning rate.

    ```python
    if not staircase:
        decayed_learning_rate = learning_rate * exp(- decay_rate * (global_step / decay_steps))
    else:
        decayed_learning_rate = learning_rate * exp(- decay_rate * (global_step / decay_steps))
    ```
    Args:
        learning_rate: A scalar float32 value or a Variable. This
          will be the initial learning rate during training
        global_step: A Variable that record the training step.
        decay_steps: A Python `int32` number.
        decay_rate: A Python `float` number.
        staircase: Boolean. If set true, decay the learning rate every decay_steps.

    Returns:
        The decayed learning rate
    """
    if not isinstance(global_step, Variable):
        raise ValueError("global_step is required for natural_exp_decay.")

    div_res = global_step / decay_steps
    if staircase:
        div_res = layers.floor(x=div_res)
    return learning_rate * layers.exp(x=(-1 * decay_rate * div_res))


def inverse_time_decay(learning_rate,
                       global_step,
                       decay_steps,
                       decay_rate,
                       staircase=False):
    """Applies inverse time decay to the initial learning rate.

    ```python
    if staircase:
      decayed_learning_rate = learning_rate / (1 + decay_rate * floor(global_step / decay_step))
    else:
      decayed_learning_rate = learning_rate / (1 + decay_rate * global_step / decay_step)
    ```
    Args:
        learning_rate: A scalar float32 value or a Variable. This
          will be the initial learning rate during training
        global_step: A Variable that record the training step.
        decay_steps: A Python `int32` number.
        decay_rate: A Python `float` number.
        staircase: Boolean. If set true, decay the learning rate every decay_steps.

    Returns:
        The decayed learning rate
    """
    if not isinstance(global_step, Variable):
        raise ValueError("global_step is required for inverse_time_decay.")

    div_res = global_step / decay_steps
    if staircase:
        div_res = layers.floor(x=div_res)

    return learning_rate / (1 + decay_rate * div_res)


def polynomial_decay(learning_rate,
                     global_step,
                     decay_steps,
                     end_learning_rate=0.0001,
                     power=1.0,
                     cycle=False):
    """Applies inverse time decay to the initial learning rate.

    ```python
    if cycle:
        decay_steps = decay_steps * ceil(global_step / decay_steps)
    else:
        global_step = min(global_step, decay_steps)
    decayed_learning_rate = (learning_rate - end_learning_rate) *
                      (1 - global_step / decay_steps) ^ power +
                      end_learning_rate
    ```
    Args:
        learning_rate: A scalar float32 value or a Variable. This
          will be the initial learning rate during training
        global_step: A Variable that record the training step.
        decay_steps: A Python `int32` number.
        end_learning_rate: A Python `float` number.
        power: A Python `float` number
        cycle: Boolean. If set true, decay the learning rate every decay_steps.

    Returns:
        The decayed learning rate
    """
    if not isinstance(global_step, Variable):
        raise ValueError("global_step is required for inverse_time_decay.")

    if cycle:
        div_res = layers.ceil(x=(global_step / decay_steps))
        zero_var = layers.fill_constant(shape=[1], dtype='float32', value=0.0)
        one_var = layers.fill_constant(shape=[1], dtype='float32', value=1.0)

        cond = layers.equal(x=global_step, y=zero_var)
        true_cond = layers.ConditionalBlock([cond])
        with true_cond.block():
            layers.assign(input=one_var, output=div_res)
        decay_steps = decay_steps * div_res
    else:
        decay_steps_var = layers.fill_constant(
            shape=[1], dtype='float32', value=float(decay_steps))
        global_step = layers.elementwise_min(x=global_step, y=decay_steps_var)

    return (learning_rate - end_learning_rate) * \
           ((1 - global_step / decay_steps) ** power) + end_learning_rate
