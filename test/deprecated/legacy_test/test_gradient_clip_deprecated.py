# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

import numpy as np
from fake_reader import fake_imdb_reader

import paddle
from paddle import base
from paddle.base import core
from paddle.nn.clip import _allow_pure_fp16_global_norm_clip

paddle.enable_static()


def bow_net(
    data, label, dict_dim, emb_dim=128, hid_dim=128, hid_dim2=96, class_dim=2
):
    """
    BOW net
    This model is from https://github.com/PaddlePaddle/models:
    base/PaddleNLP/text_classification/nets.py
    """
    emb = paddle.static.nn.embedding(
        input=data, is_sparse=True, size=[dict_dim, emb_dim]
    )
    bow = paddle.static.nn.sequence_lod.sequence_pool(
        input=emb, pool_type='sum'
    )
    bow_tanh = paddle.tanh(bow)
    fc_1 = paddle.static.nn.fc(x=bow_tanh, size=hid_dim, activation="tanh")
    fc_2 = paddle.static.nn.fc(x=fc_1, size=hid_dim2, activation="tanh")
    prediction = paddle.static.nn.fc(
        x=[fc_2], size=class_dim, activation="softmax"
    )
    cost = paddle.nn.functional.cross_entropy(
        input=prediction, label=label, reduction='none', use_softmax=False
    )
    avg_cost = paddle.mean(x=cost)

    return avg_cost


class TestGradientClip(unittest.TestCase):
    def setUp(self):
        self.word_dict_len = 5147
        self.BATCH_SIZE = 2
        reader = fake_imdb_reader(self.word_dict_len, self.BATCH_SIZE * 100)
        self.train_data = paddle.batch(reader, batch_size=self.BATCH_SIZE)
        self.clip_gradient = lambda x: None
        self.init()

    def init(self):
        pass

    def get_places(self):
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        return places

    def check_clip_result(self, out, out_clip):
        pass

    def check_gradient_clip(self, place, dtype='float32'):
        prog = base.Program()
        startup_program = base.Program()
        with base.program_guard(
            main_program=prog, startup_program=startup_program
        ):
            image = paddle.static.data(
                name="a", shape=[-1, 784], dtype='float32'
            )
            label = paddle.static.data(name="b", shape=[-1, 1], dtype='int64')
            if dtype != 'float32':
                image_cast = paddle.cast(image, dtype)
                hidden = paddle.static.nn.fc(
                    x=image_cast, size=32, activation='relu'
                )
            else:
                hidden = paddle.static.nn.fc(
                    x=image, size=32, activation='relu'
                )
            predict = paddle.static.nn.fc(
                x=hidden, size=10, activation='softmax'
            )

            cost = paddle.nn.functional.cross_entropy(
                input=predict, label=label, reduction='none', use_softmax=False
            )
            avg_cost = paddle.mean(cost)

        prog_clip = prog.clone()
        avg_cost_clip = prog_clip.block(0).var(avg_cost.name)

        p_g = base.backward.append_backward(loss=avg_cost)
        p_g_clip = base.backward.append_backward(loss=avg_cost_clip)

        p_g = sorted(p_g, key=lambda x: x[0].name)
        p_g_clip = sorted(p_g_clip, key=lambda x: x[0].name)
        with base.program_guard(
            main_program=prog_clip, startup_program=startup_program
        ):
            p_g_clip = self.clip_gradient(p_g_clip)

        grad_list = [elem[1] for elem in p_g]
        grad_clip_list = [elem[1] for elem in p_g_clip]

        train_reader = paddle.batch(paddle.dataset.mnist.train(), batch_size=3)
        exe = base.Executor(place)
        feeder = base.DataFeeder(feed_list=[image, label], place=place)
        exe.run(startup_program)

        data = next(train_reader())
        out = exe.run(prog, feed=feeder.feed(data), fetch_list=grad_list)
        out_clip = exe.run(
            prog_clip, feed=feeder.feed(data), fetch_list=grad_clip_list
        )
        self.check_clip_result(out, out_clip)

    def check_sparse_gradient_clip(self, place):
        prog = base.Program()
        startup_program = base.Program()
        with base.program_guard(
            main_program=prog, startup_program=startup_program
        ):
            data = paddle.static.data(
                name="words", shape=[-1, 1], dtype="int64", lod_level=1
            )
            label = paddle.static.data(
                name="label", shape=[-1, 1], dtype="int64"
            )
            cost = bow_net(data, label, self.word_dict_len)

            self.backward_and_optimize(cost)

        exe = base.Executor(place)
        feeder = base.DataFeeder(feed_list=[data, label], place=place)
        exe.run(startup_program)

        data = next(self.train_data())
        val = exe.run(prog, feed=feeder.feed(data), fetch_list=[cost])[0]
        self.assertEqual(val.shape, ())
        self.assertFalse(np.isnan(val))

    def backward_and_optimize(self, cost):
        pass


class TestPirGradientClipByGlobalNorm(TestGradientClip):
    def init(self):
        self.clip_norm = 0.2

    def check_clip_result(self, out, out_clip):
        global_norm = 0
        for v in out:
            global_norm += np.sum(np.square(v))
        global_norm = np.sqrt(global_norm)
        scale = self.clip_norm / np.maximum(self.clip_norm, global_norm)
        res = []
        for i in range(len(out)):
            out[i] = scale * out[i]

        for u, v in zip(out, out_clip):
            np.testing.assert_allclose(
                u,
                v,
                rtol=1e-05,
                atol=1e-08,
                err_msg=f'gradient clip by global norm has wrong results!, \nu={u}\nv={v}\ndiff={u - v}',
            )

    def _run(self, place, dtype='float32'):
        paddle.seed(2023)
        prog = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(
            main_program=prog, startup_program=startup_program
        ):
            image = paddle.static.data(
                name="a", shape=[-1, 784], dtype='float32'
            )
            label = paddle.static.data(name="b", shape=[-1, 1], dtype='int64')
            hidden_linear = paddle.nn.Linear(784, 32)
            if dtype != 'float32':
                image_cast = paddle.cast(image, dtype)
                hidden = paddle.nn.functional.relu(hidden_linear(image_cast))
            else:
                hidden = paddle.nn.functional.relu(hidden_linear(image))

            predict_linear = paddle.nn.Linear(32, 10)
            predict = paddle.nn.functional.softmax(predict_linear(hidden))

            cost = paddle.nn.functional.cross_entropy(
                input=predict, label=label, reduction='none', use_softmax=False
            )
            avg_cost = paddle.mean(cost)

            grad_list = paddle.autograd.ir_backward.grad(
                avg_cost, prog.global_block().all_parameters()
            )

            train_reader = paddle.batch(
                paddle.dataset.mnist.train(), batch_size=3
            )
            exe = base.Executor(place)
            exe.run(startup_program)
            data = next(train_reader())
            a = np.array([i[0] for i in data]).astype('float32')
            b = np.array([i[1] for i in data]).reshape(3, 1).astype('int64')
            out = exe.run(prog, feed={'a': a, 'b': b}, fetch_list=grad_list)
            return out

    def _run_clip(self, place, dtype='float32'):
        paddle.seed(2023)
        prog = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(
            main_program=prog, startup_program=startup_program
        ):
            image = paddle.static.data(
                name="a", shape=[-1, 784], dtype='float32'
            )
            label = paddle.static.data(name="b", shape=[-1, 1], dtype='int64')
            hidden_linear = paddle.nn.Linear(784, 32)
            if dtype != 'float32':
                image_cast = paddle.cast(image, dtype)
                hidden = paddle.nn.functional.relu(hidden_linear(image_cast))
            else:
                hidden = paddle.nn.functional.relu(hidden_linear(image))

            predict_linear = paddle.nn.Linear(32, 10)
            predict = paddle.nn.functional.softmax(predict_linear(hidden))

            cost = paddle.nn.functional.cross_entropy(
                input=predict, label=label, reduction='none', use_softmax=False
            )
            avg_cost = paddle.mean(cost)

            params = prog.global_block().all_parameters()
            grad_list = paddle.autograd.ir_backward.grad(avg_cost, params)

            p_g_clip = self.clip_gradient(list(zip(params, grad_list)))

            grad_clip_list = [elem[1] for elem in p_g_clip]
            train_reader = paddle.batch(
                paddle.dataset.mnist.train(), batch_size=3
            )
            exe = base.Executor(place)
            exe.run(startup_program)
            data = next(train_reader())
            a = np.array([i[0] for i in data]).astype('float32')
            b = np.array([i[1] for i in data]).reshape(3, 1).astype('int64')
            out_clip = exe.run(
                prog, feed={'a': a, 'b': b}, fetch_list=grad_clip_list
            )
            return out_clip

    def check_gradient_clip(self, place, dtype='float32'):
        out = self._run(place, dtype)
        out_clip = self._run_clip(place, dtype)
        self.check_clip_result(out, out_clip)

    def test_new_gradient_clip(self):
        def func(params_grads):
            clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=self.clip_norm)
            return clip(params_grads)

        self.clip_gradient = func
        with paddle.pir_utils.IrGuard():
            self.check_gradient_clip(base.CPUPlace())

    def check_sparse_gradient_clip(self, place):
        pass


class TestGradientClipByGlobalNorm(TestGradientClip):
    def init(self):
        self.clip_norm = 0.2

    def check_clip_result(self, out, out_clip):
        global_norm = 0
        for v in out:
            global_norm += np.sum(np.square(v))
        global_norm = np.sqrt(global_norm)
        scale = self.clip_norm / np.maximum(self.clip_norm, global_norm)
        res = []
        for i in range(len(out)):
            out[i] = scale * out[i]

        for u, v in zip(out, out_clip):
            np.testing.assert_allclose(
                u,
                v,
                rtol=1e-05,
                atol=1e-08,
                err_msg=f'gradient clip by global norm has wrong results!, \nu={u}\nv={v}\ndiff={u - v}',
            )

    # test whether the output is right when use 'set_gradient_clip'
    def test_old_gradient_clip(self):
        def func(params_grads):
            clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=self.clip_norm)
            paddle.nn.clip.set_gradient_clip(clip)
            return paddle.nn.clip.append_gradient_clip_ops(params_grads)

        self.clip_gradient = func
        self.check_gradient_clip(base.CPUPlace())

    # test whether the output is right when use grad_clip
    def test_new_gradient_clip(self):
        def func(params_grads):
            clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=self.clip_norm)
            return clip(params_grads)

        self.clip_gradient = func
        self.check_gradient_clip(base.CPUPlace())

    # test whether the output is right when use grad_clip under float64
    def test_new_gradient_clip_fp64(self):
        def func(params_grads):
            clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=self.clip_norm)
            return clip(params_grads)

        self.clip_gradient = func
        self.check_gradient_clip(base.CPUPlace(), "float64")

    # invoke 'set_gradient_clip' in a wrong order
    def test_wrong_API_order(self):
        def backward_func(cost):
            clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=5.0)
            paddle.nn.clip.set_gradient_clip(clip)
            sgd_optimizer = paddle.optimizer.SGD(
                learning_rate=0.01, grad_clip=clip
            )
            # if 'set_gradient_clip' and 'optimize(grad_clip)' together, 'set_gradient_clip' will be ineffective
            sgd_optimizer.minimize(cost)
            # 'set_gradient_clip' must before 'minimize', otherwise, 'set_gradient_clip' will be ineffective
            paddle.nn.clip.set_gradient_clip(clip)

        self.backward_and_optimize = backward_func
        for place in self.get_places():
            self.check_sparse_gradient_clip(place)

    # raise TypeError
    def test_type_error(self):
        # the type of optimizer(grad_clip=) must be an instance of GradientClipBase's derived class
        with self.assertRaises(TypeError):
            sgd_optimizer = paddle.optimizer.SGD(
                learning_rate=0.1, grad_clip="test"
            )

    # if grad is None or not need clip
    def test_none_grad_fp32(self):
        ops = self._test_none_grad_helper("float32")
        self.assertListEqual(
            ops,
            [
                'squared_l2_norm',
                'squared_l2_norm',
                'sum',
                'sqrt',
                'fill_constant',
                'elementwise_max',
                'elementwise_div',
                'elementwise_mul',
                'elementwise_mul',
            ],
        )

    def test_none_grad_fp16(self):
        ops = self._test_none_grad_helper("float16")
        self.assertListEqual(
            ops,
            [
                'squared_l2_norm',
                'squared_l2_norm',
                'sum',
                'cast',
                'sqrt',
                'fill_constant',
                'elementwise_max',
                'elementwise_div',
                'cast',
                'elementwise_mul',
                'cast',
                'elementwise_mul',
            ],
        )

    def _test_none_grad_helper(self, dtype):
        prog = base.Program()
        startup_program = base.Program()
        with base.program_guard(
            main_program=prog, startup_program=startup_program
        ):
            clip = paddle.nn.ClipGradByGlobalNorm(self.clip_norm)
            x = (
                base.default_main_program()
                .global_block()
                .create_parameter(name="x", shape=[2, 3], dtype=dtype)
            )
            y = (
                base.default_main_program()
                .global_block()
                .create_parameter(name="y", shape=[2, 3], dtype=dtype)
            )

            # (x, None) should not be returned
            params_grads = [(x, None), (x, y), (y, x)]
            params_grads = clip(params_grads)
            self.assertTrue(
                len(params_grads) == 2,
                "ClipByGlobalNorm: when grad is None, it shouldn't be returned by gradient clip!",
            )

            ops = [op.type for op in x.block.ops]
        return ops


class TestPirGradientClipByNorm(TestGradientClip):
    def init(self):
        self.clip_norm = 0.2

    def check_clip_result(self, out, out_clip):
        for u, v in zip(out, out_clip):
            norm = np.sqrt(np.sum(np.power(u, 2)))
            scale = self.clip_norm / np.maximum(self.clip_norm, norm)
            u = u * scale
            np.testing.assert_allclose(
                u,
                v,
                rtol=1e-05,
                atol=1e-08,
                err_msg='gradient clip by norm has wrong results!',
            )

    def _run(self, place, dtype='float32'):
        paddle.seed(2023)
        prog = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(
            main_program=prog, startup_program=startup_program
        ):
            image = paddle.static.data(
                name="a", shape=[-1, 784], dtype='float32'
            )
            label = paddle.static.data(name="b", shape=[-1, 1], dtype='int64')
            hidden_linear = paddle.nn.Linear(784, 32)
            if dtype != 'float32':
                image_cast = paddle.cast(image, dtype)
                hidden = paddle.nn.functional.relu(hidden_linear(image_cast))
            else:
                hidden = paddle.nn.functional.relu(hidden_linear(image))

            predict_linear = paddle.nn.Linear(32, 10)
            predict = paddle.nn.functional.softmax(predict_linear(hidden))

            cost = paddle.nn.functional.cross_entropy(
                input=predict, label=label, reduction='none', use_softmax=False
            )
            avg_cost = paddle.mean(cost)

            grad_list = paddle.autograd.ir_backward.grad(
                avg_cost, prog.global_block().all_parameters()
            )

            train_reader = paddle.batch(
                paddle.dataset.mnist.train(), batch_size=3
            )
            exe = base.Executor(place)
            exe.run(startup_program)
            data = next(train_reader())
            a = np.array([i[0] for i in data]).astype('float32')
            b = np.array([i[1] for i in data]).reshape(3, 1).astype('int64')
            out = exe.run(prog, feed={'a': a, 'b': b}, fetch_list=grad_list)
            return out

    def _run_clip(self, place, dtype='float32'):
        paddle.seed(2023)
        prog = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(
            main_program=prog, startup_program=startup_program
        ):
            image = paddle.static.data(
                name="a", shape=[-1, 784], dtype='float32'
            )
            label = paddle.static.data(name="b", shape=[-1, 1], dtype='int64')
            hidden_linear = paddle.nn.Linear(784, 32)
            if dtype != 'float32':
                image_cast = paddle.cast(image, dtype)
                hidden = paddle.nn.functional.relu(hidden_linear(image_cast))
            else:
                hidden = paddle.nn.functional.relu(hidden_linear(image))

            predict_linear = paddle.nn.Linear(32, 10)
            predict = paddle.nn.functional.softmax(predict_linear(hidden))

            cost = paddle.nn.functional.cross_entropy(
                input=predict, label=label, reduction='none', use_softmax=False
            )
            avg_cost = paddle.mean(cost)

            params = prog.global_block().all_parameters()
            grad_list = paddle.autograd.ir_backward.grad(avg_cost, params)

            p_g_clip = self.clip_gradient(list(zip(params, grad_list)))

            grad_clip_list = [elem[1] for elem in p_g_clip]
            train_reader = paddle.batch(
                paddle.dataset.mnist.train(), batch_size=3
            )
            exe = base.Executor(place)
            exe.run(startup_program)
            data = next(train_reader())
            a = np.array([i[0] for i in data]).astype('float32')
            b = np.array([i[1] for i in data]).reshape(3, 1).astype('int64')
            out_clip = exe.run(
                prog, feed={'a': a, 'b': b}, fetch_list=grad_clip_list
            )
            return out_clip

    def check_gradient_clip(self, place, dtype='float32'):
        out = self._run(place, dtype)
        out_clip = self._run_clip(place, dtype)
        self.check_clip_result(out, out_clip)

    def test_new_gradient_clip(self):
        def func(params_grads):
            clip = paddle.nn.ClipGradByNorm(clip_norm=self.clip_norm)
            return clip(params_grads)

        self.clip_gradient = func
        with paddle.pir_utils.IrGuard():
            self.check_gradient_clip(base.CPUPlace())

    def test_none_grad(self):
        clip = paddle.nn.ClipGradByNorm(self.clip_norm)
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            startup = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                x = paddle.pir.core.create_parameter(
                    dtype="float32",
                    shape=[2, 3],
                    name="x",
                    initializer=paddle.nn.initializer.Constant(value=0.5),
                    need_clip=False,
                )
                y = paddle.pir.core.create_parameter(
                    dtype="float32",
                    shape=[2, 3],
                    name="y",
                    initializer=paddle.nn.initializer.Constant(value=0.5),
                    need_clip=False,
                )
            # (x, None) should not be returned
            params_grads = [(x, None), (x, y)]
            params_grads = clip(params_grads)
            self.assertTrue(
                len(clip(params_grads)) == 1,
                "ClipGradByNorm: when grad is None, it shouldn't be returned by gradient clip!",
            )
            self.assertTrue(
                params_grads[0][1].name == 'y',
                "ClipGradByNorm: grad should not be clipped when filtered out!",
            )


class TestGradientClipByNorm(TestGradientClip):
    def init(self):
        self.clip_norm = 0.2

    def check_clip_result(self, out, out_clip):
        for u, v in zip(out, out_clip):
            norm = np.sqrt(np.sum(np.power(u, 2)))
            scale = self.clip_norm / np.maximum(self.clip_norm, norm)
            u = u * scale
            np.testing.assert_allclose(
                u,
                v,
                rtol=1e-05,
                atol=1e-08,
                err_msg='gradient clip by norm has wrong results!',
            )

    # test whether the output is right when use grad_clip
    def test_gradient_clip(self):
        def func(params_grads):
            clip = paddle.nn.ClipGradByNorm(clip_norm=self.clip_norm)
            return clip(params_grads)

        self.clip_gradient = func
        self.check_gradient_clip(base.CPUPlace())

    # if grad is None or not need clip
    def test_none_grad(self):
        clip = paddle.nn.ClipGradByNorm(self.clip_norm)
        x = (
            base.default_main_program()
            .global_block()
            .create_parameter(
                name="x", shape=[2, 3], dtype="float32", need_clip=False
            )
        )
        y = (
            base.default_main_program()
            .global_block()
            .create_parameter(
                name="y", shape=[2, 3], dtype="float32", need_clip=False
            )
        )

        # (x, None) should not be returned
        params_grads = [(x, None), (x, y)]
        params_grads = clip(params_grads)
        self.assertTrue(
            len(clip(params_grads)) == 1,
            "ClipGradByNorm: when grad is None, it shouldn't be returned by gradient clip!",
        )
        self.assertTrue(
            params_grads[0][1].name == 'y',
            "ClipGradByNorm: grad should not be clipped when filtered out!",
        )


class TestGradientClipByValue(TestGradientClip):
    def init(self):
        self.max = 0.2
        self.min = 0.1

    def check_clip_result(self, out, out_clip):
        for i, v in enumerate(out):
            out[i] = np.clip(v, self.min, self.max)
        for u, v in zip(out, out_clip):
            u = np.clip(u, self.min, self.max)
            np.testing.assert_allclose(
                u,
                v,
                rtol=1e-06,
                atol=1e-08,
                err_msg='gradient clip by value has wrong results!',
            )

    # test whether the output is right when use grad_clip
    def test_gradient_clip(self):
        def func(params_grads):
            clip = paddle.nn.ClipGradByValue(max=self.max, min=self.min)
            return clip(params_grads)

        self.clip_gradient = func
        self.check_gradient_clip(base.CPUPlace())

    # if grad is None or not need clip
    def test_none_grad(self):
        clip = paddle.nn.ClipGradByValue(self.max, self.min)
        x = (
            base.default_main_program()
            .global_block()
            .create_parameter(
                name="x", shape=[2, 3], dtype="float32", need_clip=False
            )
        )
        y = (
            base.default_main_program()
            .global_block()
            .create_parameter(
                name="y", shape=[2, 3], dtype="float32", need_clip=False
            )
        )

        # (x, None) should not be returned
        params_grads = [(x, None), (x, y)]
        params_grads = clip(params_grads)
        self.assertTrue(
            len(clip(params_grads)) == 1,
            "ClipGradByValue: when grad is None, it shouldn't be returned by gradient clip!",
        )
        self.assertTrue(
            params_grads[0][1].name == 'y',
            "ClipGradByValue: grad should not be clipped when filtered out!",
        )


class TestDygraphGradientClip(unittest.TestCase):
    def test_gradient_clip(self):
        with base.dygraph.guard():
            linear = paddle.nn.Linear(5, 5)
            inputs = paddle.uniform([16, 5], min=-10, max=10).astype('float32')
            out = linear(paddle.to_tensor(inputs))
            loss = paddle.mean(out)
            loss.backward()
            sgd_optimizer = paddle.optimizer.SGD(
                learning_rate=0.0,
                parameters=linear.parameters(),
                grad_clip=paddle.nn.ClipGradByGlobalNorm(0.1),
            )
            self.check_clip_result(loss, sgd_optimizer)

    def check_clip_result(self, loss, optimizer):
        pass


class TestDygraphGradientClipByGlobalNorm(TestDygraphGradientClip):
    def setUp(self):
        self.clip_norm = 0.8
        self.clip1 = paddle.nn.ClipGradByGlobalNorm(clip_norm=self.clip_norm)
        self.clip2 = paddle.nn.ClipGradByGlobalNorm(clip_norm=self.clip_norm)

    def check_clip_result(self, loss, optimizer):
        # if grad is None
        x = paddle.to_tensor(np.array([2, 3]).astype("float32"))
        y = paddle.to_tensor(np.array([3, 4]).astype("float32"))
        assert len(self.clip1([(x, x), (x, y), (x, None)])) == 2
        # get params and grads from network
        opt, params_grads = optimizer.minimize(loss)
        _, grads = zip(*params_grads)
        params_grads = self.clip2(params_grads)
        _, grads_clip = zip(*params_grads)

        global_norm = 0
        for u in grads:
            u = u.numpy()
            global_norm += np.sum(np.power(u, 2))
        global_norm = np.sqrt(global_norm)

        global_norm_clip = 0
        for v in grads_clip:
            v = v.numpy()
            global_norm_clip += np.sum(np.power(v, 2))
        global_norm_clip = np.sqrt(global_norm_clip)

        a = np.minimum(global_norm, self.clip_norm)
        b = global_norm_clip
        self.assertTrue(
            np.isclose(a=a, b=b, rtol=1e-6, atol=1e-8),
            f"gradient clip by global norm has wrong results, expetcd:{a:f}, but received:{b:f}",
        )


class TestDygraphGradientClipByNorm(TestDygraphGradientClip):
    def setUp(self):
        self.clip_norm = 0.8
        self.clip = paddle.nn.ClipGradByNorm(clip_norm=self.clip_norm)

    def check_clip_result(self, loss, optimizer):
        # if grad is None
        x = paddle.to_tensor(np.array([2, 3]).astype("float32"))
        assert len(self.clip([(x, None)])) == 0
        # get params and grads from network
        self.clip([(paddle.to_tensor(np.array([2, 3])), None)])
        opt, params_grads = optimizer.minimize(loss)
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
                np.isclose(a=a, b=b, rtol=1e-6, atol=1e-8),
                f"gradient clip by norm has wrong results, expetcd:{a:f}, but received:{b:f}",
            )


class TestDygraphGradientClipByValue(TestDygraphGradientClip):
    def setUp(self):
        self.max = 0.2
        self.min = 0.1
        self.clip = paddle.nn.ClipGradByValue(max=self.max, min=self.min)

    def check_clip_result(self, loss, optimizer):
        # if grad is None
        x = paddle.to_tensor(np.array([2, 3]).astype("float32"))
        assert len(self.clip([(x, None)])) == 0
        # get params and grads from network
        opt, params_grads = optimizer.minimize(loss)
        _, grads = zip(*params_grads)
        params_grads = self.clip(params_grads)
        _, grads_clip = zip(*params_grads)
        for u, v in zip(grads, grads_clip):
            u = np.clip(u.numpy(), self.min, self.max)
            v = v.numpy()
            np.testing.assert_allclose(
                u,
                v,
                rtol=1e-06,
                atol=1e-08,
                err_msg='gradient clip by value has wrong results!',
            )


class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.linear = paddle.nn.Linear(5, 5)
        self.batch_norm = paddle.nn.BatchNorm(5)

    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x)
        return x


class TestDygraphGradientClipFP16(unittest.TestCase):
    def test_gradient_clip(self):
        if base.core.is_compiled_with_cuda():
            with base.dygraph.guard():
                paddle.seed(10)
                model = SimpleNet()
                sgd_optimizer = paddle.optimizer.SGD(
                    learning_rate=0.0, parameters=model.parameters()
                )
                model, sgd_optimizer = paddle.amp.decorate(
                    models=model, optimizers=sgd_optimizer, level='O2'
                )
                scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
                inputs = paddle.uniform([1, 5], min=-10, max=10).astype(
                    'float32'
                )
                with paddle.amp.auto_cast(level='O2'):
                    out = model(paddle.to_tensor(inputs))
                    loss = paddle.mean(out)
                scaled = scaler.scale(loss)
                scaled.backward()
                scaler.unscale_(sgd_optimizer)
                # before clip
                params_grads = []
                for param in model.parameters():
                    if param.stop_gradient:
                        continue
                    if param._grad_ivar() is not None:
                        params_grads.append((param, param._grad_ivar()))
                _, grads = zip(*params_grads)
                # clip grads
                clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=0.8)
                params_grads = clip(params_grads)
                _, grads_clip = zip(*params_grads)
                # param update
                scaler.step(sgd_optimizer)
                scaler.update()

                global_norm = 0
                for u in grads:
                    u = u.numpy()
                    global_norm += np.sum(np.power(u, 2))
                global_norm = np.sqrt(global_norm)
                global_norm_clip = 0
                for v in grads_clip:
                    v = v.numpy()
                    global_norm_clip += np.sum(np.power(v, 2))
                global_norm_clip = np.sqrt(global_norm_clip)

                a = np.minimum(global_norm, 0.8)
                b = global_norm_clip
                self.assertTrue(
                    np.isclose(a=a, b=b, rtol=1e-3, atol=1e-8),
                    f"gradient clip by global norm has wrong results, expetcd:{a:f}, but received:{b:f}",
                )


class TestDygraphGradientClipFP64(unittest.TestCase):
    def test_gradient_clip(self):
        with base.dygraph.guard():
            inputs = paddle.uniform([16, 5], min=-10, max=10).astype('float32')
            linear = paddle.nn.Linear(5, 5)
            out = linear(paddle.to_tensor(inputs))
            loss = paddle.mean(out)
            loss.backward()
            # before clip
            params_grads = []
            for param in linear.parameters():
                if param.stop_gradient:
                    continue
                if param._grad_ivar() is not None:
                    params_grads.append((param, param._grad_ivar()))
            _, grads = zip(*params_grads)
            # clip grads
            clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=0.1)
            params_grads = clip(params_grads)
            _, grads_clip = zip(*params_grads)

            global_norm = 0
            for u in grads:
                u = u.numpy()
                global_norm += np.sum(np.power(u, 2))
            global_norm = np.sqrt(global_norm)

            global_norm_clip = 0
            for v in grads_clip:
                v = v.numpy()
                print(v)
                global_norm_clip += np.sum(np.power(v, 2))
            global_norm_clip = np.sqrt(global_norm_clip)
            print(global_norm_clip)

            a = np.minimum(global_norm, 0.1)
            b = global_norm_clip

            self.assertTrue(
                np.isclose(a=a, b=b, rtol=1e-6, atol=1e-8),
                f"gradient clip by global norm has wrong results, expetcd:{a:f}, but received:{b:f}",
            )


class TestPureFP16ClipGradByGlobalNorm(unittest.TestCase):
    def check_main(self, expected_has_cast_op):
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with paddle.static.program_guard(main_prog, startup_prog):
            names = ["p0", "p1"]
            shapes = [[2, 3], [4, 5]]

            param_and_grads = []
            main_block = main_prog.global_block()
            for name, shape in zip(names, shapes):
                p = main_block.create_parameter(
                    name=name, shape=shape, dtype='float16'
                )
                g = main_block.create_parameter(
                    name=p.name + '@GRAD', shape=p.shape, dtype=p.dtype
                )
                param_and_grads.append((p, g))

            clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
            clip(param_and_grads)
            actual_has_cast = any(op.type == 'cast' for op in main_block.ops)
            self.assertEqual(actual_has_cast, expected_has_cast_op)

    def test_main(self):
        self.check_main(True)
        _allow_pure_fp16_global_norm_clip(True)
        self.check_main(False)
        _allow_pure_fp16_global_norm_clip(False)
        self.check_main(True)


if __name__ == '__main__':
    unittest.main()
