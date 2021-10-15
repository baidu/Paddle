#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import contextlib
import numpy as np
from paddle import _C_ops
from paddle.fluid import core
from paddle.fluid.data_feeder import check_variable_and_dtype
from paddle.fluid.framework import in_dygraph_mode, default_main_program
from paddle.fluid.layer_helper import LayerHelper

__all__ = []

MODEL_PARALLEL_RNG = 'model_parallel_rng'

# This file is inspired by Megatron to control random states for MP:
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/mpu/random.py


class RNGStatesTracker:
    """
    Tracker the RNG states.
    """

    def __init__(self):
        # Map from name to the rng state.
        self.states_ = {}
        self.seeds_ = set()

    def reset(self):
        self.states_ = {}
        self.seeds_ = set()

    def add(self, name, seed):
        if seed in self.seeds_:
            raise ValueError('seed {} already exists'.format(seed))
        self.seeds_.add(seed)
        if name in self.states_:
            raise ValueError('state {} already exists'.format(name))
        orig_rng_state = paddle.get_cuda_rng_state()
        paddle.seed(seed)
        self.states_[name] = paddle.get_cuda_rng_state()
        paddle.set_cuda_rng_state(orig_rng_state)

    def get_states_tracker(self):
        states = {}
        for name in self.states_:
            states[name] = self.states_[name]
        return states

    def set_states_tracker(self, states):
        self.states_ = states

    @contextlib.contextmanager
    def rng_state(self, name=MODEL_PARALLEL_RNG):
        if name not in self.states_:
            raise ValueError('state {} does not exist'.format(name))
        orig_cuda_rng_state = paddle.get_cuda_rng_state()
        paddle.set_cuda_rng_state(self.states_[name])
        try:
            yield
        finally:
            self.states_[name] = paddle.get_cuda_rng_state()
            paddle.set_cuda_rng_state(orig_cuda_rng_state)


RNG_STATE_TRACKER = RNGStatesTracker()


def get_rng_state_tracker():
    return RNG_STATE_TRACKER


def model_parallel_random_seed(seed=None):
    import paddle.distributed.fleet as fleet
    hcg = fleet.get_hybrid_communicate_group()
    rank = hcg.get_model_parallel_rank()

    if seed:
        global_seed = seed
        local_seed = seed * 1024 + rank * 100
    else:
        global_seed = np.random.randint(0, 655350)
        local_seed = np.random.randint(rank * 10000, (rank + 1) * 10000 - 1)

    RNG_STATE_TRACKER.reset()
    RNG_STATE_TRACKER.add(MODEL_PARALLEL_RNG, local_seed)
    paddle.seed(global_seed)


def determinate_seed(rng_name):
    assert rng_name is not None and rng_name != ""
    helper = LayerHelper('seed', **locals())
    out = helper.create_variable_for_type_inference(dtype=paddle.int32)
    # set force_cpu to reduce sync copy from CPU->GPU->CPU, and reduce pipeline hang
    helper.append_op(
        type='seed',
        outputs={'Out': out},
        attrs={'determinate': True,
               'rng_name': rng_name,
               'force_cpu': True})
    return out


def dropout(x,
            p=0.5,
            axis=None,
            rng_name=None,
            training=True,
            mode="upscale_in_train",
            name=None):
    """
    Dropout is a regularization technique for reducing overfitting by preventing
    neuron co-adaption during training. The dropout operator randomly sets the
    outputs of some units to zero, while upscale others according to the given
    dropout probability.

    Args:
        x (Tensor): The input tensor. The data type is float32 or float64.
        p (float|int): Probability of setting units to zero. Default 0.5.
        axis (int|list|tuple): The axis along which the dropout is performed. Default None.
        training (bool): A flag indicating whether it is in train phrase or not. Default True.
        mode(str): ['upscale_in_train'(default) | 'downscale_in_infer'].

                           1. upscale_in_train(default), upscale the output at training time

                              - train: out = input * mask / ( 1.0 - dropout_prob )
                              - inference: out = input

                           2. downscale_in_infer, downscale the output at inference

                              - train: out = input * mask
                              - inference: out = input * (1.0 - dropout_prob)
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        A Tensor representing the dropout, has same shape and data type as `x` .


    Examples:
        We use ``p=0.5`` in the following description for simplicity.

        1. When ``axis=None`` , this is commonly used dropout, which dropout each element of x randomly.

        ..  code-block:: text

            Let's see a simple case when x is a 2d tensor with shape 2*3:
            [[1 2 3]
             [4 5 6]]
            we generate mask with the same shape as x, which is 2*3. The value of mask is
            sampled from a Bernoulli distribution randomly. For example, we may get such mask:
            [[0 1 0]
             [1 0 1]]
            So the output is obtained from elementwise multiply of x and mask:
            [[0 2 0]
             [4 0 6]]
            Using default setting, i.e. ``mode='upscale_in_train'`` ,
            if in training phase, the final upscale output is:
            [[0 4 0 ]
             [8 0 12]]
            if in test phase, the output is the same as input:
            [[1 2 3]
             [4 5 6]]
            we can also set ``mode='downscale_in_infer'`` , then
            if in training phase, the final output is:
            [[0 2 0]
             [4 0 6]]
            if in test phase, the scale output is:
            [[0.5 1.  1.5]
             [2.  2.5 3. ]]



        2. When ``axis!=None`` , this is useful for dropping whole channels from an image or sequence.

        ..  code-block:: text

            Let's see the simple case when x is a 2d tensor with shape 2*3 again:
            [[1 2 3]
             [4 5 6]]
            (1) If ``axis=0`` , this means the dropout is only performed in axis `0` .
                we generate mask with the shape 2*1. Only in axis `0` the value is randomly selected.
                For example, we may get such mask:
                [[1]
                 [0]]
                The output is obtained from elementwise multiply of x and mask. Doing that the mask will be
                broadcast from 2*1 to 2*3:
                [[1 1 1]
                 [0 0 0]]
                and the result after elementwise multiply is:
                [[1 2 3]
                 [0 0 0]]
                then we can do upscale or downscale according to the setting of other arguments.
            (2) If ``axis=1`` , this means the dropout is only performed in axis `1` .
                we generate mask with the shape 1*3. Only in axis `1` the value is randomly selected.
                For example, we may get such mask:
                [[1 0 1]]
                Doing elementwise multiply the mask will be broadcast from 1*3 to 2*3:
                [[1 0 1]
                 [1 0 1]]
                and the result after elementwise multiply is:
                [[1 0 3]
                 [4 0 6]]
            (3) What about ``axis=[0, 1]`` ? This means the dropout is performed in all axes of x,
                which is the same case as default setting ``axis=None`` .
            (4) You may note that logically `axis=None` means the dropout is performed in none axis of x,
                We generate mask with the shape 1*1. Whole input is randomly selected or dropped.
                For example, we may get such mask:
                [[0]]
                Doing elementwise multiply the mask will be broadcast from 1*1 to 2*3:
                [[0 0 0]
                 [0 0 0]]
                and the result after elementwise multiply is:
                [[0 0 0]
                 [0 0 0]]
                Actually this is not what we want because all elements may set to zero~

        When x is a 4d tensor with shape `NCHW`, we can set ``axis=[0,1]`` and the dropout will be performed in channel `N` and `C`, `H` and `W` is tied, i.e. paddle.nn.dropout(x, p, axis=[0,1]) . Please refer to ``paddle.nn.functional.dropout2d`` for more details.
        Similarly, when x is a 5d tensor with shape `NCDHW`, we can set ``axis=[0,1]`` to perform dropout3d. Please refer to ``paddle.nn.functional.dropout3d`` for more details.

        .. code-block:: python

            import paddle
            import numpy as np

            x = np.array([[1,2,3], [4,5,6]]).astype('float32')
            x = paddle.to_tensor(x)
            y_train = paddle.nn.functional.dropout(x, 0.5)
            y_test = paddle.nn.functional.dropout(x, 0.5, training=False)
            y_0 = paddle.nn.functional.dropout(x, axis=0)
            y_1 = paddle.nn.functional.dropout(x, axis=1)
            y_01 = paddle.nn.functional.dropout(x, axis=[0,1])
            print(x)
            print(y_train)
            print(y_test)
            print(y_0)
            print(y_1)
            print(y_01)

    """
    if rng_name is None or in_dygraph_mode():
        return paddle.nn.functional.dropout(x, p, axis, training, mode, name)

    # fast return for p == 0
    if p == 0: return x

    assert isinstance(p,
                      (float, int)), TypeError("p argument should be a number")
    assert 0 <= p <= 1, ValueError("p argument should between 0 and 1")
    assert mode in ('downscale_in_infer', 'upscale_in_train'), \
        ValueError(
            "mode argument should be 'downscale_in_infer' or 'upscale_in_train'")

    assert axis is None, \
        TypeError("unsupport axis when using random seed generator")

    mode = 'downgrade_in_infer' if mode == 'downscale_in_infer' else mode  #semantic transfer

    seed = determinate_seed(rng_name)

    helper = LayerHelper('dropout', **locals())
    check_variable_and_dtype(x, 'x', ['float16', 'float32', 'float64'],
                             'dropout')

    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    mask = helper.create_variable_for_type_inference(
        dtype=core.VarDesc.VarType.UINT8, stop_gradient=True)

    helper.append_op(
        type='dropout',
        inputs={'X': [x],
                'Seed': seed},
        outputs={'Out': [out],
                 'Mask': [mask]},
        attrs={
            'dropout_prob': p,
            'is_test': not training,
            'dropout_implementation': mode,
        })
    return out
