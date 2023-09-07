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

import unittest

import numpy as np

import paddle
import paddle.nn.functional as F
from paddle import base, static
from paddle.base import backward


class BackwardNet:
    """
    Abstract Base Class.
    All Net inherited this Class should implement two functions:
        build_model: build net to test the logic of backward
        init_data: fake input data to test all programs.
    """

    def __init__(self):
        self.stop_gradient_grad_vars = set()
        self.no_grad_vars = set()
        self.params_names = set()
        self.op_path = []

    def build_model(self):
        """
        Build net to test the logic of backward.
        :return: loss
        """
        raise NotImplementedError

    def init_data(self):
        """
        Fake input data to test all programs.
        :return: dict, {'var_name': var_data}
        """
        raise NotImplementedError


class TestBackward(unittest.TestCase):
    """
    All related TestClass should inherit this class,
    and only implement test_backward function.
    """

    def _check_all(self, net):
        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        exe = base.Executor(place)

        main = base.Program()
        startup = base.Program()

        with base.program_guard(main, startup):
            loss = net.build_model()
            self._check_backward(loss, main)

            optimizer = paddle.optimizer.SGD(learning_rate=0.1)
            optimizer.minimize(loss)
            exe.run(startup)
            exe.run(feed=net.init_data())

    def _check_backward(self, loss, main_program):
        global_block_idx = self.global_block_idx
        params_grads = self._check_params_grad(loss)
        # 1.1 get_stop_gradients
        no_grad_dict = self._check_stop_gradient(main_program)
        # 1.2 find_op_path
        op_path, block_no_grad_set = self._check_op_path(
            main_program.block(global_block_idx), [loss], [], no_grad_dict
        )
        # 1.3 _find_no_grad_vars
        no_grad_vars = self._check_find_no_grad_vars(
            main_program.block(global_block_idx),
            op_path,
            [loss],
            block_no_grad_set,
        )
        # update no_grad_dict
        block_no_grad_set.update(no_grad_vars)
        no_grad_dict[global_block_idx].update(
            list(map(base.backward._append_grad_suffix_, block_no_grad_set))
        )

    def _check_params_grad(self, loss, parameter_list=None, no_grad_set=None):
        params_grads = base.backward.append_backward(
            loss, parameter_list, no_grad_set
        )
        params_names = {
            param_var.name for (param_var, grad_var) in params_grads
        }
        self.assertSetEqual(params_names, self.net.params_names)

        return params_grads

    def _check_stop_gradient(self, program):
        no_grad_dict = base.backward._get_stop_gradients_(program)
        if no_grad_dict is not None and isinstance(no_grad_dict, dict):
            self.assertSetEqual(
                no_grad_dict[self.global_block_idx],
                self.net.stop_gradient_grad_vars,
            )

        return no_grad_dict

    def _check_op_path(self, root_block, outputs, inputs=[], no_grad_dict=None):
        if no_grad_dict is None or not isinstance(no_grad_dict, dict):
            block_no_grad_set = None
        else:
            block_no_grad_set = set(
                map(
                    base.backward._strip_grad_suffix_,
                    no_grad_dict[self.global_block_idx],
                )
            )
        op_path = base.backward._find_op_path_(
            root_block, outputs, inputs, block_no_grad_set
        )
        op_types = [op.type for op in op_path]
        self.assertListEqual(op_types, self.net.op_path)

        return op_path, block_no_grad_set

    def _check_find_no_grad_vars(
        self, root_block, op_path, targets, block_no_grad_set
    ):
        no_grad_vars = base.backward._find_no_grad_vars(
            root_block, op_path, targets, block_no_grad_set
        )
        self.assertSetEqual(no_grad_vars, self.net.no_grad_vars)

        return no_grad_vars

    def _check_error_param_list(self, net, parameter_list):
        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        exe = base.Executor(place)

        main = base.Program()
        startup = base.Program()

        with base.program_guard(main, startup):
            loss = net.build_model()
            optimizer = paddle.optimizer.SGD(learning_rate=0.1)
            optimizer.minimize(loss, parameter_list=parameter_list)
            exe.run(startup)
            exe.run(feed=net.init_data())

    def _check_error_no_grad_set(self, net, no_grad_set):
        place = (
            base.CUDAPlace(0)
            if base.core.is_compiled_with_cuda()
            else base.CPUPlace()
        )
        exe = base.Executor(place)

        main = base.Program()
        startup = base.Program()

        with base.program_guard(main, startup):
            loss = net.build_model()
            optimizer = paddle.optimizer.SGD(learning_rate=0.1)
            optimizer.minimize(loss, no_grad_set=no_grad_set)
            exe.run(startup)
            exe.run(feed=net.init_data())


class SimpleNet(BackwardNet):
    def __init__(self):
        super().__init__()
        self.stop_gradient_grad_vars = {
            'x_no_grad@GRAD',
            'x2_no_grad@GRAD',
            'x3_no_grad@GRAD',
            'label_no_grad@GRAD',
        }
        self.no_grad_vars = set()
        self.params_names = {'w2v', 'fc_predict.b_0', 'fc_w'}
        self.op_path = [
            'lookup_table_v2',
            'lookup_table_v2',  # embedding
            'elementwise_add',  # merge
            'mul',
            'elementwise_add',
            'softmax',  # fc
            'elementwise_sub',
            'square',
            'reduce_mean',
        ]  # loss
        self.shape = [16, 50]

    def init_data(self):
        assert len(self.shape) == 2
        x = np.random.randint(0, 90, self.shape).astype('int64')
        x2 = np.random.randint(0, 90, self.shape).astype('int64')
        x3 = np.random.randint(0, 90, self.shape).astype('int64')
        label = np.random.random([self.shape[0], 1]).astype('float32')
        return {
            'x_no_grad': x,
            'x2_no_grad': x2,
            'x3_no_grad': x3,
            'label_no_grad': label,
        }

    def build_model(self):
        # stop_gradient = True in input
        x = paddle.static.data(
            name='x_no_grad', shape=self.shape, dtype='int64'
        )
        x2 = paddle.static.data(
            name='x2_no_grad', shape=self.shape, dtype='int64'
        )
        x3 = paddle.static.data(
            name='x3_no_grad', shape=self.shape, dtype='int64'
        )
        label = paddle.static.data(
            name='label_no_grad', shape=[self.shape[0], 1], dtype='float32'
        )
        # shared layer, the grad of 'w2v' will be summed and renamed.
        # To test  _addup_repetitive_outputs_
        x_emb = paddle.static.nn.embedding(
            x, size=[100, 64], param_attr=base.ParamAttr(name='w2v')
        )
        x2_emb = paddle.static.nn.embedding(
            x2, size=[100, 64], param_attr=base.ParamAttr(name='w2v')
        )
        x3_emb = paddle.static.nn.embedding(
            x3, size=[100, 64], param_attr=base.ParamAttr(name='w2v')
        )
        # merge layers
        x_merge = paddle.add(x_emb, x2_emb, name='x_add_x2')
        x2_merge = paddle.add(x2_emb, x3_emb, name='x2_add_x3')
        # shared fc_w
        predict = paddle.static.nn.fc(
            x=x_merge,
            size=1,
            activation='softmax',
            weight_attr=base.ParamAttr(name='fc_w'),
            name='fc_predict',
        )
        # useless layer for calculating loss
        fc_no_use = paddle.static.nn.fc(
            x=x2_merge,
            size=1,
            activation='sigmoid',
            weight_attr=base.ParamAttr(name='fc_w'),
            name='fc_no_use',
        )
        # loss
        cost = paddle.nn.functional.square_error_cost(
            input=predict, label=label
        )
        loss = paddle.mean(cost, name='mean_loss')

        return loss


class TestSimpleNet(TestBackward):
    def test_backward(self):
        """
        Instantiate each NetClass to test backward.
        """
        self.global_block_idx = 0
        self.net = SimpleNet()
        self._check_all(self.net)


class TestGradientsError(unittest.TestCase):
    def test_error(self):
        x = paddle.static.data(name='x', shape=[None, 2, 8, 8], dtype='float32')
        x.stop_gradient = False
        conv = paddle.static.nn.conv2d(x, 4, 1, bias_attr=False)
        y = F.relu(conv)

        with self.assertRaises(TypeError):
            x_grad = base.gradients(y.name, x)

        with self.assertRaises(TypeError):
            x_grad = base.gradients(y, x.name)

        with self.assertRaises(TypeError):
            x_grad = base.gradients([y], [x], target_gradients=x.name)

        with self.assertRaises(TypeError):
            x_grad = base.gradients([y], x, no_grad_set=conv)


class TestSimpleNetWithErrorParamList(TestBackward):
    def test_parameter_list_type_error(self):
        self.global_block_idx = 0
        self.net = SimpleNet()
        # The type of parameter_list argument must be list or tuple
        with self.assertRaises(TypeError):
            self._check_error_param_list(self.net, "test")
        # The type of parameter_list's member must be Variable or str
        test = paddle.static.data(
            name='test', shape=[None, 90], dtype='float32'
        )
        with self.assertRaises(TypeError):
            self._check_error_param_list(self.net, [test, "test", 3])


class TestSimpleNetWithErrorNoGradSet(TestBackward):
    def test_no_grad_set_type_error(self):
        self.global_block_idx = 0
        self.net = SimpleNet()
        # The type of no_grad_set argument must be set or list or tuple
        with self.assertRaises(TypeError):
            self._check_error_no_grad_set(self.net, "test")
        # The type of no_grad_set's member must be Variable or str
        test = paddle.static.data(
            name='test', shape=[None, 90], dtype='float32'
        )
        with self.assertRaises(TypeError):
            self._check_error_no_grad_set(self.net, [test, "test", 3])


class TestAppendBackwardWithError(unittest.TestCase):
    def build_net(self):
        x = paddle.static.data(name='x', shape=[None, 13], dtype='int64')
        y = paddle.static.data(name='y', shape=[None, 1], dtype='float32')
        x_emb = paddle.static.nn.embedding(x, size=[100, 256])
        y_predict = paddle.static.nn.fc(x=x_emb, size=1, name='my_fc')
        loss = paddle.nn.functional.square_error_cost(input=y_predict, label=y)
        avg_loss = paddle.mean(loss)
        param_names = [
            param.name
            for param in base.default_main_program().block(0).all_parameters()
        ]

        return avg_loss, param_names

    def setUp(self):
        main_program = base.Program()
        with base.program_guard(main_program):
            self.avg_loss, self.param_names = self.build_net()

    def test_loss_type_error(self):
        with self.assertRaises(TypeError):
            base.backward.append_backward(loss=self.avg_loss.name)

    def test_parameter_list_type_error(self):
        with self.assertRaises(TypeError):
            self.param_names[0] = np.random.random([10])
            base.backward.append_backward(
                loss=self.avg_loss, parameter_list=self.param_names
            )

    def test_callback_type_error(self):
        with self.assertRaises(TypeError):

            def callback(block, context):
                return

            base.backward.append_backward(
                loss=self.avg_loss, callbacks=callback
            )


class TestGradientsWithOptimizer(unittest.TestCase):
    def _check_grad_op_name(self, forward_list, optimiezed_list):
        backward_list = [op + "_grad" for op in reversed(forward_list)]
        idx = optimiezed_list.index(backward_list[0], len(backward_list))

        self.assertListEqual(
            backward_list, optimiezed_list[idx : idx + len(backward_list)]
        )

    def test_gradient_with_optimizer(self):
        main = base.Program()
        startup = base.Program()

        with base.program_guard(main, startup):
            img = static.data(name='image', shape=[None, 784])
            pred = static.nn.fc(x=img, size=10, activation='relu')
            loss = paddle.mean(pred)
            opt = paddle.optimizer.Momentum(learning_rate=0.01, momentum=0.9)

            forward_list = [o.type for o in main.current_block().ops]
            (
                optimize_ops,
                pram_grads,
            ) = paddle.autograd.backward_mode.gradients_with_optimizer(
                main, opt
            )

            optimized_list = [o.type for o in main.current_block().ops]

            self.assertGreater(len(optimized_list), len(forward_list))
            self.assertIn(opt.type, optimized_list)
            self._check_grad_op_name(forward_list, optimized_list)


# TODO(Aurelius84): add conditional network test
class ConditionalNet(BackwardNet):
    def __init__(self):
        super().__init__()


class TestBackwardUninitializedVariable(unittest.TestCase):
    """this case is found in yolov5 while to_static.
    gradient aggregation may cause sum a invalid variable.
    """

    def test(self):
        paddle.enable_static()
        main_prg, startup_prg = paddle.static.Program(), paddle.static.Program()
        with paddle.static.program_guard(main_prg, startup_prg):
            gt = paddle.static.data(name='gt', shape=[4], dtype='float32')
            x = paddle.static.data(name='x', shape=[2], dtype='float32')
            gt.stop_gradient = True
            x.stop_gradient = False
            gt = gt.reshape([4, 1]).reshape([4])
            loss = (
                paddle.nn.functional.binary_cross_entropy(x, gt[:2])
                + (gt[2:4] * x).sum()
            )
            exe = paddle.static.Executor()
            paddle.base.backward.gradients(loss, [])
            exe.run(startup_prg)
            # Optimizer
            out = exe.run(
                main_prg,
                feed={
                    'gt': np.array([1.0, 1.0, 0.0, 0.0], dtype='float32'),
                    'x': np.array([0.5, 0.5], dtype='float32'),
                },
                fetch_list=[loss],
            )
            print(out)


class TestStripGradSuffix(unittest.TestCase):
    def test_strip_grad_suffix(self):
        cases = (
            ('x@GRAD', 'x'),
            ('x@GRAD@GRAD', 'x'),
            ('x@GRAD@RENAME@1', 'x'),
            ('x@GRAD_slice_0@GRAD', 'x@GRAD_slice_0'),
            ('grad/grad/x@GRAD@RENAME@block0@1@GRAD', 'x'),
        )
        for input_, desired in cases:
            self.assertEqual(backward._strip_grad_suffix_(input_), desired)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
