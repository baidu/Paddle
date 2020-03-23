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

from __future__ import division
from __future__ import print_function

import argparse
import contextlib
import os

import numpy as np

import paddle
from paddle import fluid
from paddle.fluid.optimizer import Momentum
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear

from model import Model, CrossEntropy, Input
from metrics import Accuracy
from distributed import prepare_context, Env, get_nranks, DistributedBatchSampler
from paddle.fluid.io import BatchSampler, DataLoader, MnistDataset

class SimpleImgConvPool(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 pool_size,
                 pool_stride,
                 pool_padding=0,
                 pool_type='max',
                 global_pooling=False,
                 conv_stride=1,
                 conv_padding=0,
                 conv_dilation=1,
                 conv_groups=None,
                 act=None,
                 use_cudnn=False,
                 param_attr=None,
                 bias_attr=None):
        super(SimpleImgConvPool, self).__init__('SimpleConv')

        self._conv2d = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=conv_stride,
            padding=conv_padding,
            dilation=conv_dilation,
            groups=conv_groups,
            param_attr=None,
            bias_attr=None,
            use_cudnn=use_cudnn)

        self._pool2d = Pool2D(
            pool_size=pool_size,
            pool_type=pool_type,
            pool_stride=pool_stride,
            pool_padding=pool_padding,
            global_pooling=global_pooling,
            use_cudnn=use_cudnn)

    def forward(self, inputs):
        x = self._conv2d(inputs)
        x = self._pool2d(x)
        return x


class MNIST(Model):
    def __init__(self):
        super(MNIST, self).__init__()
        self._simple_img_conv_pool_1 = SimpleImgConvPool(
            1, 20, 5, 2, 2, act="relu")

        self._simple_img_conv_pool_2 = SimpleImgConvPool(
            20, 50, 5, 2, 2, act="relu")

        pool_2_shape = 50 * 4 * 4
        SIZE = 10
        scale = (2.0 / (pool_2_shape**2 * SIZE))**0.5
        self._fc = Linear(
            800,
            10,
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.NormalInitializer(
                    loc=0.0, scale=scale)),
            act="softmax")

    def forward(self, inputs):
        inputs = fluid.layers.reshape(inputs, [-1, 1, 28, 28])
        x = self._simple_img_conv_pool_1(inputs)
        x = self._simple_img_conv_pool_2(x)
        x = fluid.layers.flatten(x, axis=1)
        x = self._fc(x)
        return x


class CustromMnistDataset(MnistDataset):
    def __init__(self,
                 image_filename=None,
                 label_filename=None,
                 mode='train',
                 download=True):
        super(CustromMnistDataset, self).__init__(image_filename, 
                                    label_filename, mode, download)


    def __getitem__(self, idx):
        return self.images[idx], [self.labels[idx]]


def main():
    @contextlib.contextmanager
    def null_guard():
        yield

    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id) \
        if fluid.dygraph.parallel.Env().nranks > 1 else fluid.CUDAPlace(0)
    guard = fluid.dygraph.guard(place) if FLAGS.dynamic else null_guard()
    if fluid.dygraph.parallel.Env().nranks > 1:
        prepare_context(place)


    if not os.path.exists('mnist_checkpoints'):
        os.mkdir('mnist_checkpoints')

    with guard:
        train_dataset = CustromMnistDataset(mode='train')
        val_dataset = CustromMnistDataset(mode='test')
        
        inputs = [Input([None, 784], 'float32', name='image')]
        labels = [Input([None, 1], 'int64', name='label')]

        if fluid.in_dygraph_mode():
            feed_list = None
        else:
            feed_list = [x.forward() for x in inputs + labels]
            
        if get_nranks() > 1:
            train_sampler = DistributedBatchSampler(train_dataset, batch_size=FLAGS.batch_size, shuffle=True)
            train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, places=place, 
                                    feed_list=feed_list, num_workers=4, return_list=True)
            val_sampler = DistributedBatchSampler(val_dataset, batch_size=FLAGS.batch_size)
            val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, places=place, 
                                    feed_list=feed_list, num_workers=4, return_list=True)
        else:
            train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, places=place, 
                                    feed_list=feed_list, num_workers=4, return_list=True)
            val_loader = DataLoader(val_dataset, batch_size=FLAGS.batch_size, places=place, 
                                    feed_list=feed_list, num_workers=4, return_list=True)
                                    
        model = MNIST()
        optim = Momentum(
            learning_rate=FLAGS.lr,
            momentum=.9,
            parameter_list=model.parameters())
        
        model.prepare(optim, CrossEntropy(), Accuracy(topk=(1, 2)), inputs, labels)
        if FLAGS.resume is not None:
            model.load(FLAGS.resume)

        model.fit(train_loader, val_loader, epochs=FLAGS.epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("CNN training on MNIST")
    parser.add_argument(
        "-d", "--dynamic", action='store_true', help="enable dygraph mode")
    parser.add_argument(
        "-e", "--epoch", default=100, type=int, help="number of epoch")
    parser.add_argument(
        '--lr',
        '--learning-rate',
        default=1e-3,
        type=float,
        metavar='LR',
        help='initial learning rate')
    parser.add_argument(
        "-b", "--batch_size", default=128, type=int, help="batch size")
    parser.add_argument(
        "-n", "--num_devices", default=1, type=int, help="number of devices")
    parser.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="checkpoint path to resume")
    FLAGS = parser.parse_args()
    main()
