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

from __future__ import print_function

import unittest
import numpy as np
import os
import sys
import six

import paddle
import paddle.nn as nn
import paddle.optimizer as opt
import paddle.fluid as fluid
from paddle.fluid.optimizer import Adam
import paddle.fluid.framework as framework
from test_imperative_base import new_program_scope

BATCH_SIZE = 16
BATCH_NUM = 4
EPOCH_NUM = 4
SEED = 10

IMAGE_SIZE = 784
CLASS_NUM = 10

if six.PY2:
    LARGE_PARAM = 2**2
else:
    LARGE_PARAM = 2**26


def random_batch_reader():
    def _get_random_inputs_and_labels():
        np.random.seed(SEED)
        image = np.random.random([BATCH_SIZE, IMAGE_SIZE]).astype('float32')
        label = np.random.randint(0, CLASS_NUM - 1, (
            BATCH_SIZE,
            1, )).astype('int64')
        return image, label

    def __reader__():
        for _ in range(BATCH_NUM):
            batch_image, batch_label = _get_random_inputs_and_labels()
            batch_image = paddle.to_tensor(batch_image)
            batch_label = paddle.to_tensor(batch_label)
            yield batch_image, batch_label

    return __reader__


class LinearNet(nn.Layer):
    def __init__(self):
        super(LinearNet, self).__init__()
        self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

    def forward(self, x):
        return self._linear(x)


class LayerWithLargeParameters(paddle.nn.Layer):
    def __init__(self):
        super(LayerWithLargeParameters, self).__init__()
        self._l = paddle.nn.Linear(10, LARGE_PARAM)

    def forward(self, x):
        y = self._l(x)
        return y


def train(layer, loader, loss_fn, opt):
    for epoch_id in range(EPOCH_NUM):
        for batch_id, (image, label) in enumerate(loader()):
            out = layer(image)
            loss = loss_fn(out, label)
            loss.backward()
            opt.step()
            opt.clear_grad()


class TestSaveLoadLargeParameters(unittest.TestCase):
    def setUp(self):
        pass

    def test_large_parameters_paddle_save(self):
        # enable dygraph mode
        paddle.disable_static()
        # create network
        layer = LayerWithLargeParameters()
        save_dict = layer.state_dict()

        path = os.path.join("test_paddle_save_load_large_param_save",
                            "layer.pdparams")
        if six.PY2:
            protocol = 2
        else:
            protocol = 4
        paddle.save(save_dict, path, protocol=protocol)
        dict_load = paddle.load(path)
        # compare results before and after saving
        for key, value in save_dict.items():
            self.assertTrue(
                np.array_equal(dict_load[key].numpy(), value.numpy()))


class TestSaveLoadPickle(unittest.TestCase):
    def test_pickle_protocol(self):
        # enable dygraph mode
        paddle.disable_static()
        # create network
        layer = LinearNet()
        save_dict = layer.state_dict()

        path = os.path.join("test_paddle_save_load_pickle_protocol",
                            "layer.pdparams")

        with self.assertRaises(ValueError):
            paddle.save(save_dict, path, 2.0)

        with self.assertRaises(ValueError):
            paddle.save(save_dict, path, 1)

        with self.assertRaises(ValueError):
            paddle.save(save_dict, path, 5)

        protocols = [2, ]
        if sys.version_info.major >= 3 and sys.version_info.minor >= 4:
            protocols += [3, 4]
        for protocol in protocols:
            paddle.save(save_dict, path, pickle_protocol=protocol)
            dict_load = paddle.load(path)
            # compare results before and after saving
            for key, value in save_dict.items():
                self.assertTrue(
                    np.array_equal(dict_load[key].numpy(), value.numpy()))


class TestSaveLoadAny(unittest.TestCase):
    def set_zero(self, prog, place, scope=None):
        if scope is None:
            scope = fluid.global_scope()
        for var in prog.list_vars():
            if isinstance(var, framework.Parameter) or var.persistable:
                ten = scope.find_var(var.name).get_tensor()
                if ten is not None:
                    ten.set(np.zeros_like(np.array(ten)), place)
                    new_t = np.array(scope.find_var(var.name).get_tensor())
                    self.assertTrue(np.sum(np.abs(new_t)) == 0)

    def replace_static_save(self, program, model_path, pickle_protocol=2):
        with self.assertRaises(TypeError):
            program.state_dict(1)
        with self.assertRaises(TypeError):
            program.state_dict(scope=1)
        with self.assertRaises(ValueError):
            program.state_dict('x')
        state_dict_param = program.state_dict('param')
        paddle.save(state_dict_param, model_path + '.pdparams')
        state_dict_opt = program.state_dict('opt')
        paddle.save(state_dict_opt, model_path + '.pdopt')
        state_dict_all = program.state_dict()
        paddle.save(state_dict_opt, model_path + '.pdall')

    def replace_static_load(self, program, model_path):
        with self.assertRaises(TypeError):
            program.set_state_dict(1)
        state_dict_param = paddle.load(model_path + '.pdparams')
        state_dict_param['fake_var_name.@@'] = np.random.randn(1, 2)
        state_dict_param['static_x'] = 'UserWarning'
        program.set_state_dict(state_dict_param)
        state_dict_param['static_x'] = np.random.randn(1, 2)
        program.set_state_dict(state_dict_param)
        program.set_state_dict(state_dict_param)
        state_dict_opt = paddle.load(model_path + '.pdopt')
        program.set_state_dict(state_dict_opt)

    def test_replace_static_save_load(self):
        paddle.enable_static()
        with new_program_scope():
            x = paddle.static.data(
                name="static_x", shape=[None, IMAGE_SIZE], dtype='float32')
            z = paddle.static.nn.fc(x, 10)
            z = paddle.static.nn.fc(z, 10, bias_attr=False)
            loss = fluid.layers.reduce_mean(z)
            opt = Adam(learning_rate=1e-3)
            opt.minimize(loss)
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(paddle.static.default_startup_program())
            prog = paddle.static.default_main_program()
            fake_inputs = np.random.randn(2, IMAGE_SIZE).astype('float32')
            exe.run(prog, feed={'static_x': fake_inputs}, fetch_list=[loss])
            base_map = {}
            for var in prog.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    t = np.array(fluid.global_scope().find_var(var.name)
                                 .get_tensor())
                    base_map[var.name] = t
            path = os.path.join("test_replace_static_save_load", "model")
            # paddle.save, legacy paddle.fluid.load
            self.replace_static_save(prog, path)
            # set var to zero
            self.set_zero(prog, place)
            paddle.fluid.io.load(prog, path)
            for var in prog.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    new_t = np.array(fluid.global_scope().find_var(var.name)
                                     .get_tensor())
                    base_t = base_map[var.name]
                    self.assertTrue(np.array_equal(new_t, np.array(base_t)))
            # legacy paddle.fluid.save, paddle.load 
            paddle.fluid.io.save(prog, path)
            # set var to zero
            self.set_zero(prog, place)
            self.replace_static_load(prog, path)
            for var in prog.list_vars():
                if isinstance(var, framework.Parameter) or var.persistable:
                    new_t = np.array(fluid.global_scope().find_var(var.name)
                                     .get_tensor())
                    base_t = base_map[var.name]
                    self.assertTrue(np.array_equal(new_t, base_t))
            # test for return tensor
            path_vars = 'test_replace_save_load_return_tensor_static/model'
            for var in prog.list_vars():
                if var.persistable:
                    tensor = var.get_value(fluid.global_scope())
                    paddle.save(tensor, os.path.join(path_vars, var.name))
            with self.assertRaises(TypeError):
                var.get_value('fluid.global_scope()')
            with self.assertRaises(ValueError):
                x.get_value()
            with self.assertRaises(TypeError):
                x.set_value('1')
            fake_data = np.zeros([3, 2, 1, 2, 3])
            with self.assertRaises(TypeError):
                x.set_value(fake_data, '1')
            with self.assertRaises(ValueError):
                x.set_value(fake_data)
            with self.assertRaises(ValueError):
                var.set_value(fake_data)
            # set var to zero
            self.set_zero(prog, place)
            for var in prog.list_vars():
                if var.persistable:
                    tensor = paddle.load(
                        os.path.join(path_vars, var.name), return_numpy=False)
                    var.set_value(tensor)
                    new_t = np.array(fluid.global_scope().find_var(var.name)
                                     .get_tensor())
                    base_t = base_map[var.name]
                    self.assertTrue(np.array_equal(new_t, base_t))

    def test_paddle_save_load_v2(self):
        paddle.disable_static()
        layer = LinearNet()
        state_dict = layer.state_dict()
        path = 'paddle_save_load_v2/model.pdparams'
        with self.assertRaises(TypeError):
            paddle.save(state_dict, path, use_binary_format='False')
        # legacy paddle.save, paddle.load
        paddle.framework.io._legacy_save(state_dict, path)
        load_dict_tensor = paddle.load(path, return_numpy=False)
        # legacy paddle.load, paddle.save
        paddle.save(state_dict, path)
        load_dict_np = paddle.framework.io._legacy_load(path)
        for k, v in state_dict.items():
            self.assertTrue(
                np.array_equal(v.numpy(), load_dict_tensor[k].numpy()))
            self.assertTrue(np.array_equal(v.numpy(), load_dict_np[k]))

    def test_single_pickle_var_dygraph(self):
        # enable dygraph mode
        paddle.disable_static()
        layer = LinearNet()
        path = 'paddle_save_load_v2/var_dygraph'
        tensor = layer._linear.weight
        with self.assertRaises(ValueError):
            paddle.save(tensor, path, pickle_protocol='3')
        with self.assertRaises(ValueError):
            paddle.save(tensor, path, pickle_protocol=5)
        paddle.save(tensor, path)
        t_dygraph = paddle.load(path)
        np_dygraph = paddle.load(path, return_numpy=True)
        self.assertTrue(isinstance(t_dygraph, paddle.fluid.core.VarBase))
        self.assertTrue(np.array_equal(tensor.numpy(), np_dygraph))
        self.assertTrue(np.array_equal(tensor.numpy(), t_dygraph.numpy()))
        paddle.enable_static()
        lod_static = paddle.load(path)
        np_static = paddle.load(path, return_numpy=True)
        self.assertTrue(isinstance(lod_static, paddle.fluid.core.LoDTensor))
        self.assertTrue(np.array_equal(tensor.numpy(), np_static))
        self.assertTrue(np.array_equal(tensor.numpy(), np.array(lod_static)))

    def test_single_pickle_var_static(self):
        # enable static mode
        paddle.enable_static()
        with new_program_scope():
            # create network
            x = paddle.static.data(
                name="x", shape=[None, IMAGE_SIZE], dtype='float32')
            z = paddle.static.nn.fc(x, 128)
            loss = fluid.layers.reduce_mean(z)
            place = fluid.CPUPlace(
            ) if not paddle.fluid.core.is_compiled_with_cuda(
            ) else fluid.CUDAPlace(0)
            exe = paddle.static.Executor(place)
            exe.run(paddle.static.default_startup_program())
            prog = paddle.static.default_main_program()
            for var in prog.list_vars():
                if list(var.shape) == [IMAGE_SIZE, 128]:
                    tensor = var.get_value()
                    break
            scope = fluid.global_scope()
        origin_tensor = np.array(tensor)
        path = 'test_single_pickle_var_static/var'
        paddle.save(tensor, path)
        self.set_zero(prog, place, scope)
        # static load
        lod_static = paddle.load(path)
        np_static = paddle.load(path, return_numpy=True)
        # set_tensor(np.ndarray)
        var.set_value(np_static, scope)
        self.assertTrue(np.array_equal(origin_tensor, np.array(tensor)))
        # set_tensor(LoDTensor)
        self.set_zero(prog, place, scope)
        var.set_value(lod_static, scope)
        self.assertTrue(np.array_equal(origin_tensor, np.array(tensor)))
        # enable dygraph mode
        paddle.disable_static()
        var_dygraph = paddle.load(path)
        np_dygraph = paddle.load(path, return_numpy=True)
        self.assertTrue(np.array_equal(np.array(tensor), np_dygraph))
        self.assertTrue(np.array_equal(np.array(tensor), var_dygraph.numpy()))

    def test_dygraph_save_static_load(self):
        inps = np.random.randn(1, IMAGE_SIZE).astype('float32')
        path = 'test_dygraph_save_static_load/dy-static.pdparams'
        paddle.disable_static()
        with paddle.utils.unique_name.guard():
            layer = LinearNet()
            state_dict_dy = layer.state_dict()
            paddle.save(state_dict_dy, path)
        paddle.enable_static()
        with new_program_scope():
            layer = LinearNet()
            data = paddle.static.data(
                name='x_static_save', shape=(None, IMAGE_SIZE), dtype='float32')
            y_static = layer(data)
            program = paddle.static.default_main_program()
            place = fluid.CPUPlace(
            ) if not paddle.fluid.core.is_compiled_with_cuda(
            ) else fluid.CUDAPlace(0)
            exe = paddle.static.Executor(paddle.CPUPlace())
            exe.run(paddle.static.default_startup_program())
            state_dict = paddle.load(path, keep_name_table=True)
            program.set_state_dict(state_dict)
            state_dict_param = program.state_dict("param")
            for name, tensor in state_dict_dy.items():
                self.assertTrue(
                    np.array_equal(tensor.numpy(),
                                   np.array(state_dict_param[tensor.name])))


class TestSaveLoad(unittest.TestCase):
    def setUp(self):
        # enable dygraph mode
        paddle.disable_static()

        # config seed
        paddle.seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)

    def build_and_train_model(self):
        # create network
        layer = LinearNet()
        loss_fn = nn.CrossEntropyLoss()

        adam = opt.Adam(learning_rate=0.001, parameters=layer.parameters())

        # create data loader
        # TODO: using new DataLoader cause unknown Timeout on windows, replace it
        loader = random_batch_reader()

        # train
        train(layer, loader, loss_fn, adam)

        return layer, adam

    def check_load_state_dict(self, orig_dict, load_dict):
        for var_name, value in orig_dict.items():
            load_value = load_dict[var_name].numpy() if hasattr(
                load_dict[var_name], 'numpy') else np.array(load_dict[var_name])
            self.assertTrue(np.array_equal(value.numpy(), load_value))

    def test_save_load(self):
        layer, opt = self.build_and_train_model()

        # save
        layer_save_path = "test_paddle_save_load.linear.pdparams"
        opt_save_path = "test_paddle_save_load.linear.pdopt"
        layer_state_dict = layer.state_dict()
        opt_state_dict = opt.state_dict()

        paddle.save(layer_state_dict, layer_save_path)
        paddle.save(opt_state_dict, opt_save_path)

        # load
        load_layer_state_dict = paddle.load(layer_save_path)
        load_opt_state_dict = paddle.load(opt_save_path)

        self.check_load_state_dict(layer_state_dict, load_layer_state_dict)
        self.check_load_state_dict(opt_state_dict, load_opt_state_dict)

        # test save load in static mode
        paddle.enable_static()
        static_save_path = "static_mode_test/test_paddle_save_load.linear.pdparams"
        paddle.save(layer_state_dict, static_save_path)
        load_static_state_dict = paddle.load(static_save_path)
        self.check_load_state_dict(layer_state_dict, load_static_state_dict)

        # error test cases, some tests relay base test above
        # 1. test save obj not dict error
        test_list = [1, 2, 3]
        with self.assertRaises(NotImplementedError):
            paddle.save(test_list, "not_dict_error_path")

        # 2. test save path format error
        with self.assertRaises(ValueError):
            paddle.save(layer_state_dict, "test_paddle_save_load.linear.model/")

        # 3. test load path not exist error
        with self.assertRaises(ValueError):
            paddle.load("test_paddle_save_load.linear.params")

        # 4. test load old save path error
        with self.assertRaises(ValueError):
            paddle.load("test_paddle_save_load.linear")


if __name__ == '__main__':
    unittest.main()
