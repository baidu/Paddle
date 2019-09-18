# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
import paddle.fluid as fluid
fluid.core._set_eager_deletion_mode(-1, -1, False)

import paddle.fluid.layers.ops as ops
from paddle.fluid.initializer import init_on_cpu
from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter
from simple_nets import init_data
import math
import os
os.environ['CPU_NUM'] = str(4)

# FIXME(zcd): If the neural net has dropout_op, the output of ParallelExecutor
# and Executor is different. Because, for ParallelExecutor, the dropout_op of
# the neural net will be copied N copies(N is the number of device). This will
# lead to the random numbers generated by ParallelExecutor and Executor are different.
# So, if we compare the loss of ParallelExecutor and Executor, we should remove the
# dropout_op.
remove_dropout = False

# FIXME(zcd): If the neural net has batch_norm, the output of ParallelExecutor
# and Executor is different.
remove_bn = False

remove_dropout = True
remove_bn = True


def squeeze_excitation(input, num_channels, reduction_ratio):
    # pool = fluid.layers.pool2d(
    #    input=input, pool_size=0, pool_type='avg', global_pooling=True)
    conv = input
    shape = conv.shape
    reshape = fluid.layers.reshape(
        x=conv, shape=[-1, shape[1], shape[2] * shape[3]])
    pool = fluid.layers.reduce_mean(input=reshape, dim=2)

    squeeze = fluid.layers.fc(input=pool,
                              size=num_channels // reduction_ratio,
                              act='relu')
    excitation = fluid.layers.fc(input=squeeze,
                                 size=num_channels,
                                 act='sigmoid')
    scale = fluid.layers.elementwise_mul(x=input, y=excitation, axis=0)
    return scale


def conv_bn_layer(input, num_filters, filter_size, stride=1, groups=1,
                  act=None):
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=(filter_size - 1) // 2,
        groups=groups,
        act=None,
        bias_attr=False)
    return conv if remove_bn else fluid.layers.batch_norm(
        input=conv, act=act, momentum=0.1)


def shortcut(input, ch_out, stride):
    ch_in = input.shape[1]
    if ch_in != ch_out:
        if stride == 1:
            filter_size = 1
        else:
            filter_size = 3
        return conv_bn_layer(input, ch_out, filter_size, stride)
    else:
        return input


def bottleneck_block(input, num_filters, stride, cardinality, reduction_ratio):
    # The number of first 1x1 convolutional channels for each bottleneck build block
    # was halved to reduce the compution cost.
    conv0 = conv_bn_layer(
        input=input, num_filters=num_filters, filter_size=1, act='relu')
    conv1 = conv_bn_layer(
        input=conv0,
        num_filters=num_filters * 2,
        filter_size=3,
        stride=stride,
        groups=cardinality,
        act='relu')
    conv2 = conv_bn_layer(
        input=conv1, num_filters=num_filters * 2, filter_size=1, act=None)
    scale = squeeze_excitation(
        input=conv2,
        num_channels=num_filters * 2,
        reduction_ratio=reduction_ratio)

    short = shortcut(input, num_filters * 2, stride)

    return fluid.layers.elementwise_add(x=short, y=scale, act='relu')


img_shape = [3, 224, 224]


def SE_ResNeXt50Small(use_feed):

    img = fluid.layers.data(name='image', shape=img_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    conv = conv_bn_layer(
        input=img, num_filters=16, filter_size=3, stride=2, act='relu')
    conv = conv_bn_layer(
        input=conv, num_filters=16, filter_size=3, stride=1, act='relu')
    conv = conv_bn_layer(
        input=conv, num_filters=16, filter_size=3, stride=1, act='relu')
    conv = fluid.layers.pool2d(
        input=conv, pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')

    cardinality = 32
    reduction_ratio = 16
    depth = [3, 4, 6, 3]
    num_filters = [128, 256, 512, 1024]

    for block in range(len(depth)):
        for i in range(depth[block]):
            conv = bottleneck_block(
                input=conv,
                num_filters=num_filters[block],
                stride=2 if i == 0 and block != 0 else 1,
                cardinality=cardinality,
                reduction_ratio=reduction_ratio)

    shape = conv.shape
    reshape = fluid.layers.reshape(
        x=conv, shape=[-1, shape[1], shape[2] * shape[3]])
    pool = fluid.layers.reduce_mean(input=reshape, dim=2)
    dropout = pool if remove_dropout else fluid.layers.dropout(
        x=pool, dropout_prob=0.2, seed=1)
    # Classifier layer:
    prediction = fluid.layers.fc(input=dropout, size=1000, act='softmax')
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    loss = fluid.layers.mean(loss)
    return loss


def cosine_decay(learning_rate, step_each_epoch, epochs=120):
    """
    Applies cosine decay to the learning rate.
    lr = 0.05 * (math.cos(epoch * (math.pi / 120)) + 1)
    """
    global_step = _decay_step_counter()

    with init_on_cpu():
        epoch = ops.floor(global_step / step_each_epoch)
        decayed_lr = learning_rate * \
                     (ops.cos(epoch * (math.pi / epochs)) + 1)/2
    return decayed_lr


def optimizer(learning_rate=0.01):
    optimizer = fluid.optimizer.Momentum(
        learning_rate=cosine_decay(
            learning_rate=learning_rate, step_each_epoch=2, epochs=1),
        momentum=0.9,
        regularization=fluid.regularizer.L2Decay(1e-4))
    return optimizer


model = SE_ResNeXt50Small


def batch_size():
    return 12


def iter(use_cuda):
    if use_cuda:
        return 10
    return 2


gpu_img, gpu_label = init_data(
    batch_size=batch_size(), img_shape=img_shape, label_range=999)
cpu_img, cpu_label = init_data(
    batch_size=batch_size(), img_shape=img_shape, label_range=999)
feed_dict_gpu = {"image": gpu_img, "label": gpu_label}
feed_dict_cpu = {"image": cpu_img, "label": cpu_label}


def feed_dict(use_cuda):
    if use_cuda:
        return feed_dict_gpu
    return feed_dict_cpu
