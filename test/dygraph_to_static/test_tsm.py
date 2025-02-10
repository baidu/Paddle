#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import random
import sys
import unittest

import numpy as np
from dygraph_to_static_utils import (
    Dy2StTestBase,
    enable_to_static_guard,
    test_default_and_pir,
)
from tsm_config_utils import merge_configs, parse_config, print_configs

import paddle
from paddle.nn import BatchNorm, Linear

random.seed(0)
np.random.seed(0)


def parse_args():
    parser = argparse.ArgumentParser("Paddle Video train script")
    parser.add_argument(
        '--config',
        type=str,
        default='tsm.yaml',
        help='path to config file of model',
    )
    parser.add_argument(
        '--use_gpu',
        type=bool,
        default=paddle.is_compiled_with_cuda(),
        help='default use gpu.',
    )
    args = parser.parse_args(
        ['--config', os.path.join(os.path.dirname(__file__), 'tsm.yaml')]
    )
    return args


class ConvBNLayer(paddle.nn.Layer):
    def __init__(
        self,
        num_channels,
        num_filters,
        filter_size,
        stride=1,
        groups=1,
        act=None,
    ):
        super().__init__()

        self._conv = paddle.nn.Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=1,
            weight_attr=paddle.ParamAttr(),
            bias_attr=False,
        )

        self._batch_norm = BatchNorm(
            num_filters,
            act=act,
            param_attr=paddle.ParamAttr(),
            bias_attr=paddle.ParamAttr(),
        )

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)

        return y


class BottleneckBlock(paddle.nn.Layer):
    def __init__(
        self, num_channels, num_filters, stride, shortcut=True, seg_num=8
    ):
        super().__init__()

        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act='relu',
        )
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu',
        )
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None,
        )

        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 4,
                filter_size=1,
                stride=stride,
            )
        self.shortcut = shortcut
        self.seg_num = seg_num
        self._num_channels_out = int(num_filters * 4)

    def forward(self, inputs):
        shifts = paddle.nn.functional.temporal_shift(
            inputs, self.seg_num, 1.0 / 8
        )
        y = self.conv0(shifts)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = paddle.nn.functional.relu(paddle.add(x=short, y=conv2))
        return y


class TSM_ResNet(paddle.nn.Layer):
    def __init__(self, name_scope, config, mode):
        super().__init__(name_scope)

        self.layers = config.MODEL.num_layers
        self.seg_num = config.MODEL.seg_num
        self.class_dim = config.MODEL.num_classes
        self.reshape_list = [
            config.MODEL.seglen * 3,
            config[mode.upper()]['target_size'],
            config[mode.upper()]['target_size'],
        ]

        if self.layers == 50:
            depth = [3, 4, 6, 3]
        else:
            raise NotImplementedError
        num_filters = [64, 128, 256, 512]

        self.conv = ConvBNLayer(
            num_channels=3, num_filters=64, filter_size=7, stride=2, act='relu'
        )
        self.pool2d_max = paddle.nn.MaxPool2D(
            kernel_size=3, stride=2, padding=1
        )

        self.bottleneck_block_list = []
        num_channels = 64

        for block in range(len(depth)):
            shortcut = False
            for i in range(depth[block]):
                bottleneck_block = self.add_sublayer(
                    f'bb_{block}_{i}',
                    BottleneckBlock(
                        num_channels=num_channels,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        shortcut=shortcut,
                        seg_num=self.seg_num,
                    ),
                )
                num_channels = int(bottleneck_block._num_channels_out)
                self.bottleneck_block_list.append(bottleneck_block)
                shortcut = True
        self.pool2d_avg = paddle.nn.AdaptiveAvgPool2D(1)
        import math

        stdv = 1.0 / math.sqrt(2048 * 1.0)

        self.out = Linear(
            2048,
            self.class_dim,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(-stdv, stdv)
            ),
            bias_attr=paddle.ParamAttr(
                learning_rate=2.0, regularizer=paddle.regularizer.L1Decay()
            ),
        )

    def forward(self, inputs):
        y = paddle.reshape(inputs, [-1, *self.reshape_list])
        y = self.conv(y)
        y = self.pool2d_max(y)
        for bottleneck_block in self.bottleneck_block_list:
            y = bottleneck_block(y)
        y = self.pool2d_avg(y)
        y = paddle.nn.functional.dropout(y, p=0.5)
        y = paddle.reshape(y, [-1, self.seg_num, y.shape[1]])
        y = paddle.mean(y, axis=1)
        y = paddle.reshape(y, shape=[-1, 2048])
        y = self.out(y)
        y = paddle.nn.functional.softmax(y)
        return y


class FakeDataReader:
    def __init__(self, mode, cfg):
        self.format = cfg.MODEL.format
        self.num_classes = cfg.MODEL.num_classes
        self.seg_num = cfg.MODEL.seg_num
        self.seglen = cfg.MODEL.seglen

        self.target_size = cfg[mode.upper()]['target_size']
        self.img_mean = (
            np.array(cfg.MODEL.image_mean).reshape([3, 1, 1]).astype(np.float32)
        )
        self.img_std = (
            np.array(cfg.MODEL.image_std).reshape([3, 1, 1]).astype(np.float32)
        )

        self.batch_size = (
            1
            if sys.platform == 'darwin' or os.name == 'nt'
            else cfg[mode.upper()]['batch_size']
        )
        self.generator_out = []
        self.total_iter = 3
        for i in range(self.total_iter):
            batch_out = []
            for j in range(self.batch_size):
                label = np.int64(random.randint(0, self.num_classes - 1))
                random_mean = self.img_mean[0][0][0]
                random_std = self.img_std[0][0][0]
                imgs = np.random.normal(
                    random_mean,
                    random_std,
                    [
                        self.seg_num,
                        self.seglen * 3,
                        self.target_size,
                        self.target_size,
                    ],
                ).astype(np.float32)
                batch_out.append((imgs, label))
            self.generator_out.append(batch_out)

    def create_reader(self):
        def batch_reader():
            for i in range(self.total_iter):
                yield self.generator_out[i]

        return batch_reader


def create_optimizer(cfg, params):
    total_videos = cfg.total_videos
    batch_size = (
        1 if sys.platform == 'darwin' or os.name == 'nt' else cfg.batch_size
    )
    step = int(total_videos / batch_size + 1)
    bd = [e * step for e in cfg.decay_epochs]
    base_lr = cfg.learning_rate
    lr_decay = cfg.learning_rate_decay
    lr = [base_lr, base_lr * lr_decay, base_lr * lr_decay * lr_decay]
    l2_weight_decay = cfg.l2_weight_decay
    momentum = cfg.momentum

    optimizer = paddle.optimizer.Momentum(
        learning_rate=paddle.optimizer.lr.PiecewiseDecay(
            boundaries=bd, values=lr
        ),
        momentum=momentum,
        weight_decay=paddle.regularizer.L2Decay(l2_weight_decay),
        parameters=params,
    )

    return optimizer


def train(args, fake_data_reader):
    config = parse_config(args.config)
    train_config = merge_configs(config, 'train', vars(args))
    valid_config = merge_configs(config, 'valid', vars(args))
    print_configs(train_config, 'Train')

    random.seed(0)
    np.random.seed(0)
    paddle.seed(1000)
    paddle.framework.random._manual_program_seed(1000)

    video_model = paddle.jit.to_static(TSM_ResNet("TSM", train_config, 'Train'))

    optimizer = create_optimizer(train_config.TRAIN, video_model.parameters())

    train_reader = fake_data_reader.create_reader()

    ret = []
    for epoch in range(train_config.TRAIN.epoch):
        video_model.train()
        total_loss = 0.0
        total_acc1 = 0.0
        total_acc5 = 0.0
        total_sample = 0
        for batch_id, data in enumerate(train_reader()):
            x_data = np.array([item[0] for item in data])
            y_data = np.array([item[1] for item in data]).reshape([-1, 1])

            imgs = paddle.to_tensor(x_data)
            labels = paddle.to_tensor(y_data)
            labels.stop_gradient = True
            outputs = video_model(imgs)
            loss = paddle.nn.functional.cross_entropy(
                input=outputs,
                label=labels,
                ignore_index=-1,
                reduction='none',
                use_softmax=False,
            )
            avg_loss = paddle.mean(loss)
            acc_top1 = paddle.static.accuracy(input=outputs, label=labels, k=1)
            acc_top5 = paddle.static.accuracy(input=outputs, label=labels, k=5)

            avg_loss.backward()
            optimizer.minimize(avg_loss)
            video_model.clear_gradients()

            total_loss += float(avg_loss)
            total_acc1 += float(acc_top1)
            total_acc5 += float(acc_top5)
            total_sample += 1

            print(
                f'TRAIN Epoch {epoch}, iter {batch_id}, loss = {float(avg_loss)}, acc1 {float(acc_top1)}, acc5 {float(acc_top5)}'
            )
            ret.extend(
                [
                    float(avg_loss),
                    float(acc_top1),
                    float(acc_top5),
                ]
            )

        print(
            f'TRAIN End, Epoch {epoch}, avg_loss= {total_loss / total_sample}, avg_acc1= {total_acc1 / total_sample}, avg_acc5= {total_acc5 / total_sample}'
        )
    return ret


class TestTsm(Dy2StTestBase):
    @test_default_and_pir
    def test_dygraph_static_same_loss(self):
        if paddle.is_compiled_with_cuda():
            paddle.set_flags({"FLAGS_cudnn_deterministic": True})
        args = parse_args()
        fake_data_reader = FakeDataReader("train", parse_config(args.config))
        with enable_to_static_guard(False):
            dygraph_loss = train(args, fake_data_reader)

        static_loss = train(args, fake_data_reader)
        np.testing.assert_allclose(dygraph_loss, static_loss, rtol=1e-05)


if __name__ == '__main__':
    unittest.main()
