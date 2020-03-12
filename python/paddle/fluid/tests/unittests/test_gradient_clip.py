#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
import six
from fake_reader import fake_imdb_reader


def bow_net(words,
            label,
            dict_dim,
            emb_dim=128,
            hid_dim=128,
            hid_dim2=96,
            class_dim=2):
    """
    BOW net
    This model is from https://github.com/PaddlePaddle/models:
    fluid/PaddleNLP/text_classification/nets.py
    """
    emb = fluid.layers.embedding(
        input=words, is_sparse=True, size=[dict_dim, emb_dim])
    bow = fluid.layers.sequence_pool(input=emb, pool_type='sum')
    bow_tanh = fluid.layers.tanh(bow)
    fc_1 = fluid.layers.fc(input=bow_tanh, size=hid_dim, act="tanh")
    fc_2 = fluid.layers.fc(input=fc_1, size=hid_dim2, act="tanh")
    prediction = fluid.layers.fc(input=[fc_2], size=class_dim, act="softmax")
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)

    return avg_cost


class TestGradientClip(unittest.TestCase):
    def setUp(self):
        self.word_dict_len = 5147
        self.BATCH_SIZE = 2
        reader = fake_imdb_reader(self.word_dict_len, self.BATCH_SIZE * 100)
        self.train_data = paddle.batch(reader, batch_size=self.BATCH_SIZE)
        self.init()

    def init(self):
        pass

    def get_places(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        return places

    def gradient_clip(self, params_grads):
        pass

    def check_output(self, out, out_clip):
        pass

    def check_gradient_clip(self, place):
        prog = fluid.framework.Program()
        startup_program = fluid.framework.Program()
        with fluid.program_guard(
                main_program=prog, startup_program=startup_program):
            image = fluid.data(name='x', shape=[-1, 784], dtype='float32')
            label = fluid.data(name='y', shape=[-1, 1], dtype='int64')

            hidden1 = fluid.layers.fc(input=image, size=128, act='relu')
            hidden2 = fluid.layers.fc(input=hidden1, size=64, act='relu')
            predict = fluid.layers.fc(input=hidden2, size=10, act='softmax')

            cost = fluid.layers.cross_entropy(input=predict, label=label)
            avg_cost = fluid.layers.mean(cost)

        prog_clip = prog.clone()
        avg_cost_clip = prog_clip.block(0).var(avg_cost.name)

        p_g = fluid.backward.append_backward(loss=avg_cost)
        p_g_clip = fluid.backward.append_backward(loss=avg_cost_clip)

        p_g = sorted(p_g, key=lambda x: x[0].name)
        p_g_clip = sorted(p_g_clip, key=lambda x: x[0].name)
        with fluid.program_guard(
                main_program=prog_clip, startup_program=startup_program):
            p_g_clip = self.gradient_clip(p_g_clip)

        grad_list = [elem[1] for elem in p_g]
        grad_clip_list = [elem[1] for elem in p_g_clip]

        train_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.mnist.train(), buf_size=8192),
            batch_size=128)

        exe = fluid.Executor(place)
        feeder = fluid.DataFeeder(feed_list=[image, label], place=place)
        exe.run(startup_program)

        data = next(train_reader())
        out = exe.run(prog, feed=feeder.feed(data), fetch_list=grad_list)
        out_clip = exe.run(prog_clip,
                           feed=feeder.feed(data),
                           fetch_list=grad_clip_list)

    def backward_and_optimize(cost):
        pass

    def check_with_optimize(self, place):
        prog = fluid.framework.Program()
        startup_program = fluid.framework.Program()
        with fluid.program_guard(
                main_program=prog, startup_program=startup_program):
            words = fluid.data(
                name="words", shape=[-1, 1], dtype="int64", lod_level=1)
            label = fluid.data(name="label", shape=[-1, 1], dtype="int64")
            cost = bow_net(words, label, self.word_dict_len)
            self.backward_and_optimize(cost)

        exe = fluid.Executor(place)
        feeder = fluid.DataFeeder(feed_list=[words, label], place=place)
        exe.run(startup_program)

        data = next(self.train_data())
        val = exe.run(prog, feed=feeder.feed(data), fetch_list=[cost])[0]
        self.assertEqual((1, ), val.shape, "the shape of loss is not [1]")
        self.assertFalse(np.isnan(val), "the loss of network is nan")


class TestGradientClipByGlobalNorm(TestGradientClip):
    def init(self):
        self.clip_norm = 1.0

    def gradient_clip(self, params_grads):
        return fluid.grad_clip_global_norm(
            params_grads, clip_norm=self.clip_norm)

    def check_output(self, out, out_clip):
        global_norm = 0
        for v in out:
            global_norm += np.sum(np.power(v, 2))
        global_norm = np.sqrt(global_norm)

        global_norm_clip = 0
        for v in out_clip:
            global_norm_clip += np.sum(np.power(v, 2))
        global_norm_clip = np.sqrt(global_norm_clip)

        a = np.minimum(global_norm, self.clip_norm)
        b = global_norm_clip
        self.assertTrue(
            np.isclose(
                a=a, b=b, rtol=5e-3),
            "gradient clip by global norm has wrong results, expetcd:%f, but recieved:%f"
            % (a, b))

    def test_gradient_clip(self):
        self.check_gradient_clip(core.CPUPlace())

    def test_gradient_clip_with_optimize_1(self):
        def func(cost):
            sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.1)
            params_grads = sgd_optimizer.backward(cost)
            params_grads = fluid.grad_clip_global_norm(
                params_grads, clip_norm=self.clip_norm)
            sgd_optimizer.apply_optimize(params_grads)

        self.backward_and_optimize = func
        for place in self.get_places():
            self.check_with_optimize(place)

    def test_gradient_clip_with_optimize_2(self):
        def func(cost):
            sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.1)
            clip = fluid.clip.GradClipByGlobalNorm(clip_norm=self.clip_norm)
            sgd_optimizer.minimize(cost, grad_clip=clip)

        self.backward_and_optimize = func
        for place in self.get_places():
            self.check_with_optimize(place)


class TestGradientClipByNorm(TestGradientClip):
    def init(self):
        self.clip_norm = 1.0

    def gradient_clip(self, params_grads):
        return fluid.grad_clip_norm(params_grads, clip_norm=self.clip_norm)

    def check_output(self, out, out_clip):
        self.assertTrue(
            len(out) == len(out_clip),
            "Number of gradient changed after clipping, before clip:%d, after clip:%d"
            % (len(out), len(out_clip)))
        for u, v in zip(out, out_clip):
            a = np.sqrt(np.sum(np.power(u, 2)))
            a = np.minimum(a, self.clip_norm)
            b = np.sqrt(np.sum(np.power(v, 2)))
            self.assertTrue(
                np.isclose(
                    a=a, b=b, rtol=5e-3),
                "gradient clip by norm has wrong results, expetcd:%f, but recieved:%f"
                % (a, b))

    def test_gradient_clip(self):
        self.check_gradient_clip(core.CPUPlace())

    def test_gradient_clip_with_optimize_1(self):
        def func(cost):
            sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.1)
            params_grads = sgd_optimizer.backward(cost)
            params_grads = fluid.grad_clip_norm(
                params_grads, clip_norm=self.clip_norm)
            sgd_optimizer.apply_optimize(params_grads)

        self.backward_and_optimize = func
        for place in self.get_places():
            self.check_with_optimize(place)

    def test_gradient_clip_with_optimize_2(self):
        def func(cost):
            sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.1)
            clip = fluid.clip.GradClipByNorm(clip_norm=self.clip_norm)
            sgd_optimizer.minimize(cost, grad_clip=clip)

        self.backward_and_optimize = func
        for place in self.get_places():
            self.check_with_optimize(place)


class TestGradientClipByValue(TestGradientClip):
    def init(self):
        self.max = 1.0
        self.min = 0.1

    def gradient_clip(self, params_grads):
        return fluid.grad_clip_value(params_grads, max=self.max, min=self.min)

    def check_output(self, out, out_clip):
        for i, v in enumerate(out):
            out[i] = np.clip(v, self.min, self.max)

        self.assertTrue(
            len(out) == len(out_clip),
            "Number of gradient changed after clipping, before clip:%d, after clip:%d"
            % (len(out), len(out_clip)))
        for u, v in zip(out, out_clip):
            self.assertTrue(
                np.allclose(
                    a=u, b=v, rtol=5e-3),
                "gradient clip by value has wrong results")

    def test_gradient_clip(self):
        self.check_gradient_clip(core.CPUPlace())

    def test_gradient_clip_with_optimize_1(self):
        def func(cost):
            sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.1)
            params_grads = sgd_optimizer.backward(cost)
            params_grads = fluid.grad_clip_value(
                params_grads, max=self.max, min=self.min)
            sgd_optimizer.apply_optimize(params_grads)

        self.backward_and_optimize = func
        for place in self.get_places():
            self.check_with_optimize(place)

    def test_gradient_clip_with_optimize_2(self):
        def func(cost):
            sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.1)
            clip = fluid.clip.GradClipByValue(
                min_value=self.min, max_value=self.max)
            sgd_optimizer.minimize(cost, grad_clip=clip)

        self.backward_and_optimize = func
        for place in self.get_places():
            self.check_with_optimize(place)


class TestDygraphGradientClip(unittest.TestCase):
    def setUp(self):
        self.clip_norm = 1.0
        self.clip = fluid.clip.GradClipByGlobalNorm(clip_norm=self.clip_norm)

    def test_gradient_clip(self):
        with fluid.dygraph.guard():
            linear = fluid.dygraph.Linear(10, 10)
            inputs = fluid.layers.uniform_random([32, 10]).astype('float32')
            out = linear(fluid.dygraph.to_variable(inputs))
            loss = fluid.layers.reduce_mean(out)
            loss.backward()
            sgd_optimizer = fluid.optimizer.SGD(
                learning_rate=0.1, parameter_list=linear.parameters())
            self.check_output(loss, sgd_optimizer)

    def check_output(self, loss, optimizer):
        optimizer.minimize(loss, grad_clip=self.clip)
        self.assertEqual((1, ),
                         loss.numpy().shape, "the shape of loss is not [1]")
        self.assertFalse(
            np.isnan(loss.numpy()), "the loss of dygraph network is nan")


class TestDygraphGradientClipByGlobalNorm(TestDygraphGradientClip):
    def check_output(self, loss, optimizer):
        params_grads = optimizer.backward(loss)
        _, grads = zip(*params_grads)
        params_grads = self.clip(params_grads)
        _, grads_clip = zip(*params_grads)

        global_norm = 0
        for v in grads:
            v = v.numpy()
            global_norm += np.sum(np.power(v, 2))
        global_norm = np.sqrt(global_norm)

        global_norm_clip = 0
        for v in grads_clip:
            v = v.numpy()
            global_norm_clip += np.sum(np.power(v, 2))
        global_norm_clip = np.sqrt(global_norm_clip)

        a = np.minimum(global_norm, self.clip_norm)
        b = global_norm_clip
        self.assertTrue(
            np.isclose(
                a=a, b=b, rtol=5e-3),
            "gradient clip by global norm has wrong results, expetcd:%f, but recieved:%f"
            % (a, b))


class TestDygraphGradientClipByNorm(TestDygraphGradientClip):
    def setUp(self):
        self.clip_norm = 1.0
        self.clip = fluid.clip.GradClipByNorm(clip_norm=self.clip_norm)

    def check_output(self, loss, optimizer):
        params_grads = optimizer.backward(loss)
        _, grads = zip(*params_grads)
        params_grads = self.clip(params_grads)
        _, grads_clip = zip(*params_grads)

        for u, v in zip(grads, grads_clip):
            u = u.numpy()
            v = v.numpy()
            a = np.sqrt(np.sum(np.power(u, 2)))
            a = np.minimum(a, self.clip_norm)
            b = np.sqrt(np.sum(np.power(v, 2)))
            self.assertTrue(
                np.isclose(
                    a=a, b=b, rtol=5e-3),
                "gradient clip by norm has wrong results, expetcd:%f, but recieved:%f"
                % (a, b))


class TestDygraphGradientClipByValue(TestDygraphGradientClip):
    def setUp(self):
        self.max = 1.0
        self.min = 0.1
        self.clip = fluid.clip.GradClipByValue(
            min_value=self.min, max_value=self.max)

    def check_output(self, loss, optimizer):
        params_grads = optimizer.backward(loss)
        _, grads = zip(*params_grads)
        params_grads = self.clip(params_grads)
        _, grads_clip = zip(*params_grads)

        for u, v in zip(grads, grads_clip):
            a = np.clip(u.numpy(), self.min, self.max)
            b = v.numpy()
            self.assertTrue(
                np.allclose(
                    a=a, b=b, rtol=5e-3),
                "gradient clip by value has wrong results")


if __name__ == '__main__':
    unittest.main()
