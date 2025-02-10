#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import tempfile
import unittest

import numpy as np
from test_imperative_resnet import ResNet, optimizer_setting, train_parameters

import paddle
from paddle import base, nn
from paddle.autograd import PyLayer
from paddle.static import InputSpec

if base.core.is_compiled_with_cuda():
    base.set_flags({"FLAGS_cudnn_deterministic": True})


class SimpleConv(paddle.nn.Layer):
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
            groups=groups,
            bias_attr=None,
        )

    def forward(self, inputs):
        return self._conv(inputs)


class TestAutoCast(unittest.TestCase):
    def amp_guard_white_op(self):
        data = np.random.uniform(-1, 1, [10, 3, 32, 32]).astype('float32')
        with base.dygraph.guard():
            conv2d = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
            data = paddle.to_tensor(data)
            with paddle.amp.amp_guard(True):
                out_fp16 = conv2d(data)

            with paddle.amp.amp_guard(False):
                out_fp32 = conv2d(data)

        self.assertTrue(data.dtype == paddle.float32)
        self.assertTrue(out_fp16.dtype == paddle.float16)
        self.assertTrue(out_fp32.dtype == paddle.float32)

    def test_amp_guard_white_op(self):
        self.amp_guard_white_op()

    def amp_guard_black_op(self):
        data = np.random.uniform(-1, 1, [10, 3, 32, 32]).astype('float32')
        with base.dygraph.guard():
            data = paddle.to_tensor(data)
            with paddle.amp.amp_guard(True):
                out_fp32 = paddle.mean(data)

        self.assertTrue(data.dtype == paddle.float32)
        self.assertTrue(out_fp32.dtype == paddle.float32)

    def test_amp_guard_black_op(self):
        self.amp_guard_black_op()

    def custom_op_list(self):
        with base.dygraph.guard():
            tracer = base.framework._dygraph_tracer()
            base_white_list = paddle.amp.white_list()["float16"]["O1"]
            base_black_list = paddle.amp.black_list()["float16"]["O1"]
            with paddle.amp.amp_guard(
                custom_white_list=["log"], custom_black_list=["conv2d"]
            ):
                white_list, black_list = tracer._get_amp_op_list()
                self.assertTrue(
                    set(white_list)
                    == (set(base_white_list) | {"log"}) - {"conv2d"}
                )

                self.assertTrue(
                    set(black_list)
                    == (set(base_black_list) - {"log"}) | {"conv2d"}
                )

            base_white_list = paddle.amp.white_list()["float16"]["O2"]
            base_black_list = paddle.amp.black_list()["float16"]["O2"]
            with paddle.amp.amp_guard(
                custom_white_list=["log"],
                custom_black_list=["conv2d"],
                level='O2',
            ):
                white_list, black_list = tracer._get_amp_op_list()
                self.assertTrue(
                    set(white_list)
                    == (set(base_white_list) | {"log"}) - {"conv2d"}
                )

                self.assertTrue(
                    set(black_list)
                    == (set(base_black_list) - {"log"}) | {"conv2d"}
                )

    def test_custom_op_list(self):
        self.custom_op_list()

    def custom_op_list_exception(self):
        inp_np = np.random.random(size=[1, 3, 128, 128]).astype(np.float32)

        def func():
            with base.dygraph.guard():
                model = SimpleConv(
                    num_channels=3,
                    num_filters=64,
                    filter_size=7,
                    stride=2,
                    act='relu',
                )
                with paddle.amp.amp_guard(
                    custom_white_list=["conv2d"], custom_black_list=["conv2d"]
                ):
                    inp = paddle.to_tensor(inp_np)
                    out = model(inp)

        self.assertRaises(ValueError, func)

    def test_custom_op_list_exception(self):
        self.custom_op_list_exception()

    def amp_guard_unsupported_fp16_op(self):
        data = np.random.uniform(-1, 1, [10, 3, 32, 32]).astype('float32')
        with base.dygraph.guard():
            conv2d = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
            data = paddle.to_tensor(data)
            with paddle.amp.amp_guard(True):
                out_amp_fp16 = conv2d(data)
                out_amp_fp32 = paddle.expand_as(
                    out_amp_fp16, out_amp_fp16
                )  # expand_as_v2 has no fp16 kernel

            with paddle.amp.amp_guard(True, level='O2'):
                out_purefp16_fp16 = conv2d(data)
                out_purefp16_fp32 = paddle.expand_as(
                    out_purefp16_fp16, out_purefp16_fp16
                )  # expand_as_v2 has no fp16 kernel
        self.assertTrue(data.dtype == paddle.float32)
        self.assertTrue(out_amp_fp16.dtype == paddle.float16)
        self.assertTrue(out_amp_fp32.dtype == paddle.float32)
        self.assertTrue(out_purefp16_fp16.dtype == paddle.float16)
        self.assertTrue(out_purefp16_fp32.dtype == paddle.float32)

    def test_amp_guard_unsupported_fp16_op(self):
        self.amp_guard_unsupported_fp16_op()

    def mode_exception(self):
        def func():
            data = np.random.uniform(-1, 1, [10, 3, 32, 32]).astype('float32')
            with base.dygraph.guard():
                conv2d = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
                data = paddle.to_tensor(data)
                with paddle.amp.amp_guard(level='O'):
                    out = conv2d(data)

        self.assertRaises(ValueError, func)

    def test_mode_exception(self):
        self.mode_exception()


class TestAmpScaler(unittest.TestCase):
    def scale(self):
        if not paddle.amp.is_float16_supported():
            return
        with base.dygraph.guard():
            with paddle.amp.auto_cast(dtype='float16'):
                data = paddle.rand([10, 1024])
            scaler = paddle.amp.AmpScaler(init_loss_scaling=1024)
            scaled_data = scaler.scale(data)
            np.testing.assert_array_equal(
                scaled_data.numpy(), data.numpy() * 1024
            )

    def test_scale(self):
        self.scale()

    def minimize(self):
        inp_np = np.random.random(size=[1, 3, 128, 128]).astype(np.float32)

        def run_simple_conv(inp_np, use_scaler=True):
            paddle.seed(10)
            paddle.framework.random._manual_program_seed(10)
            with base.dygraph.guard():
                model = SimpleConv(
                    num_channels=3,
                    num_filters=64,
                    filter_size=7,
                    stride=2,
                    act='relu',
                )
                optimizer = paddle.optimizer.SGD(
                    learning_rate=0.01, parameters=model.parameters()
                )
                scaler = paddle.amp.AmpScaler(init_loss_scaling=1024)
                data = paddle.to_tensor(inp_np)

                out = model(data)
                loss = paddle.mean(out)
                if use_scaler:
                    print('use scaler')
                    scaled_loss = scaler.scale(loss)
                    scaled_loss.backward()
                    optimize_ops, params_grads = scaler.minimize(
                        optimizer, scaled_loss
                    )
                else:
                    print('use no scaler')
                    loss.backward()
                    optimize_ops, params_grads = optimizer.minimize(loss)
            return optimize_ops, params_grads

        outs_with_scaler = run_simple_conv(inp_np, use_scaler=True)
        outs_no_scaler = run_simple_conv(inp_np, use_scaler=False)

        self.assertEqual(
            outs_with_scaler[0], []
        )  # optimize_ops is [] in dygraph mode
        self.assertEqual(
            outs_no_scaler[0], []
        )  # optimize_ops is [] in dygraph mode
        for i in range(len(outs_with_scaler[1])):
            # check each grad
            np.testing.assert_allclose(
                outs_with_scaler[1][i][1].numpy(),
                outs_no_scaler[1][i][1].numpy(),
                rtol=1e-05,
            )
            # check each parameter
            np.testing.assert_allclose(
                outs_with_scaler[1][i][0].numpy(),
                outs_no_scaler[1][i][0].numpy(),
                rtol=1e-05,
            )

    def test_minimize(self):
        self.minimize()

    def step(self):
        inp_np = np.random.random(size=[1, 3, 128, 128]).astype(np.float32)

        def run_simple_conv(inp_np, use_scaler=True):
            paddle.seed(10)
            paddle.framework.random._manual_program_seed(10)
            with base.dygraph.guard():
                model = SimpleConv(
                    num_channels=3,
                    num_filters=64,
                    filter_size=7,
                    stride=2,
                    act='relu',
                )
                optimizer = paddle.optimizer.SGD(
                    learning_rate=0.01, parameters=model.parameters()
                )
                scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
                data = paddle.to_tensor(inp_np)

                out = model(data)
                loss = paddle.mean(out)
                if use_scaler:
                    print('use scaler')
                    scaled_loss = scaler.scale(loss)
                    scaled_loss.backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    print('use no scaler')
                    loss.backward()
                    optimizer.step()
            return optimizer._parameter_list

        outs_with_scaler = run_simple_conv(inp_np, use_scaler=True)
        outs_no_scaler = run_simple_conv(inp_np, use_scaler=False)

        for i in range(len(outs_with_scaler)):
            # check each parameter
            np.testing.assert_allclose(
                outs_with_scaler[i].numpy(),
                outs_no_scaler[i].numpy(),
                rtol=1e-05,
            )

    def test_step(self):
        self.step()

    def nan_inf(self):
        inp_np = np.random.random(size=[1, 3, 128, 128]).astype(np.float32)
        inp_np[0][1][2][3] = np.nan
        with base.dygraph.guard():
            model = SimpleConv(
                num_channels=3,
                num_filters=64,
                filter_size=7,
                stride=2,
                act='relu',
            )
            params_init = {}
            for param in model.parameters():
                params_init[param.name] = param.numpy()
            optimizer = paddle.optimizer.SGD(
                learning_rate=0.01, parameters=model.parameters()
            )
            scaler = paddle.amp.AmpScaler(init_loss_scaling=1024)
            data = paddle.to_tensor(inp_np)
            with paddle.amp.auto_cast(dtype='float16'):
                out = model(data)
                loss = paddle.mean(out)
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()
            optimize_ops, params_grads = scaler.minimize(optimizer, scaled_loss)
            self.assertEqual(scaler._found_inf.numpy() >= 1, True)

            for param in model.parameters():
                # param not update when tensor contains nan or inf
                np.testing.assert_array_equal(
                    param.numpy(), params_init[param.name]
                )

    def test_nan_inf(self):
        if not paddle.amp.is_float16_supported():
            return
        self.nan_inf()

    def step_update_exception(self):
        def func1():
            model = paddle.nn.Conv2D(3, 2, 3, bias_attr=True)
            optimizer = paddle.optimizer.SGD(
                learning_rate=0.01, parameters=model.parameters()
            )
            scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
            data = paddle.rand([10, 3, 32, 32])
            conv = model(data)
            loss = paddle.mean(conv)
            scaled = scaler.scale(loss)
            scaled.backward()
            scaler.unscale_(optimizer)
            scaler.unscale_(optimizer)

        self.assertRaises(RuntimeError, func1)

        def func2():
            model = paddle.nn.Conv2D(3, 2, 3, bias_attr=True)
            optimizer = paddle.optimizer.SGD(
                learning_rate=0.01, parameters=model.parameters()
            )
            scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
            data = paddle.rand([10, 3, 32, 32])
            conv = model(data)
            loss = paddle.mean(conv)
            scaled = scaler.scale(loss)
            scaled.backward()
            scaler.step(optimizer)
            scaler.unscale_(optimizer)

        self.assertRaises(RuntimeError, func2)

        def func3():
            model = paddle.nn.Conv2D(3, 2, 3, bias_attr=True)
            optimizer = paddle.optimizer.SGD(
                learning_rate=0.01, parameters=model.parameters()
            )
            scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
            data = paddle.rand([10, 3, 32, 32])
            conv = model(data)
            loss = paddle.mean(conv)
            scaled = scaler.scale(loss)
            scaled.backward()
            scaler.step(optimizer)
            scaler.step(optimizer)

        self.assertRaises(RuntimeError, func3)

    def test_step_update_exception(self):
        self.step_update_exception()

    def test_get_and_set(self):
        with base.dygraph.guard():
            scaler = paddle.amp.GradScaler(
                enable=True,
                init_loss_scaling=1024,
                incr_ratio=2.0,
                decr_ratio=0.5,
                incr_every_n_steps=1000,
                decr_every_n_nan_or_inf=2,
                use_dynamic_loss_scaling=True,
            )
            self.assertEqual(scaler.is_enable(), True)
            self.assertEqual(scaler.get_init_loss_scaling() == 1024, True)
            self.assertEqual(scaler.get_incr_ratio() == 2.0, True)
            self.assertEqual(scaler.get_decr_ratio() == 0.5, True)
            self.assertEqual(scaler.get_incr_every_n_steps() == 1000, True)
            self.assertEqual(scaler.get_decr_every_n_nan_or_inf() == 2, True)
            self.assertEqual(scaler.is_use_dynamic_loss_scaling(), True)
            scaler.set_decr_every_n_nan_or_inf(4)
            self.assertEqual(scaler.get_decr_every_n_nan_or_inf() == 4, True)
            scaler.set_decr_ratio(0.1)
            self.assertEqual(scaler.get_decr_ratio() == 0.1, True)
            scaler.set_incr_every_n_steps(200)
            self.assertEqual(scaler.get_incr_every_n_steps() == 200, True)
            scaler.set_incr_ratio(3.0)
            self.assertEqual(scaler.get_incr_ratio() == 3.0, True)
            scaler.set_init_loss_scaling(100)
            self.assertEqual(scaler.get_init_loss_scaling() == 100, True)

    def test_state_dict_and_load_state_dict(self):
        with base.dygraph.guard():
            scaler1 = paddle.amp.GradScaler(
                enable=True,
                init_loss_scaling=14,
                incr_ratio=233.0,
                decr_ratio=0.523,
                incr_every_n_steps=1090,
                decr_every_n_nan_or_inf=20,
                use_dynamic_loss_scaling=True,
            )
            scaler_state = scaler1.state_dict()
            scaler2 = paddle.amp.GradScaler(enable=True)
            scaler2.load_state_dict(scaler_state)
            self.assertEqual(scaler2.get_init_loss_scaling() == 14, True)
            self.assertEqual(scaler2.get_incr_ratio() == 233.0, True)
            self.assertEqual(scaler2.get_decr_ratio() == 0.523, True)
            self.assertEqual(scaler2.get_incr_every_n_steps() == 1090, True)
            self.assertEqual(scaler2.get_decr_every_n_nan_or_inf() == 20, True)

            scaler3 = paddle.amp.GradScaler(enable=False)
            scaler3.load_state_dict(scaler_state)
            self.assertFalse(scaler3.is_enable())

    def test_state_dict_and_load_state_dict_error(self):
        def test_error():
            state_empty = {}
            scaler = paddle.amp.GradScaler(enable=True)
            scaler.load_state_dict(state_empty)

        self.assertRaises(RuntimeError, test_error)


def reader_decorator(reader):
    def __reader__():
        for item in reader():
            img = np.array(item[0]).astype('float32').reshape(3, 224, 224)
            label = np.array(item[1]).astype('int64').reshape(1)
            yield img, label

    return __reader__


class TestGradScalerStateDict(unittest.TestCase):
    def train_resnet(
        self, enable_amp=True, use_data_loader=True, use_save_load=True
    ):
        seed = 90

        batch_size = train_parameters["batch_size"]
        batch_num = 4

        paddle.seed(seed)
        paddle.framework.random._manual_program_seed(seed)

        resnet = ResNet(use_cudnn=True)
        optimizer = optimizer_setting(
            train_parameters, parameter_list=resnet.parameters()
        )
        np.random.seed(seed)
        train_reader = paddle.batch(
            paddle.dataset.flowers.train(use_xmap=False), batch_size=batch_size
        )

        dy_param_init_value = {}
        for param in resnet.parameters():
            dy_param_init_value[param.name] = param.numpy()

        program = None
        scaler = paddle.amp.GradScaler(
            enable=enable_amp, init_loss_scaling=2.0**10
        )

        if use_data_loader:
            train_reader = paddle.batch(
                reader_decorator(paddle.dataset.flowers.train(use_xmap=False)),
                batch_size=batch_size,
                drop_last=True,
            )
            train_loader = base.io.DataLoader.from_generator(
                capacity=4,
                use_double_buffer=True,
                iterable=True,
                return_list=True,
            )
            train_loader.set_sample_list_generator(train_reader)
            train_reader = train_loader

        for batch_id, data in enumerate(train_reader()):
            if batch_id >= batch_num:
                break
            if use_data_loader:
                img, label = data
            else:
                dy_x_data = np.array(
                    [x[0].reshape(3, 224, 224) for x in data]
                ).astype('float32')
                if (
                    len(np.array([x[1] for x in data]).astype('int64'))
                    != batch_size
                ):
                    continue
                y_data = (
                    np.array([x[1] for x in data])
                    .astype('int64')
                    .reshape(-1, 1)
                )

                img = paddle.to_tensor(dy_x_data)
                label = paddle.to_tensor(y_data)
            label.stop_gradient = True

            with paddle.amp.auto_cast(enable=enable_amp):
                out = resnet(img)

            loss = paddle.nn.functional.cross_entropy(input=out, label=label)
            avg_loss = paddle.mean(x=loss)

            dy_out = avg_loss.numpy()

            scaled_loss = scaler.scale(avg_loss)
            scaled_loss.backward()

            scaler.minimize(optimizer, scaled_loss)

            dy_grad_value = {}
            for param in resnet.parameters():
                if param.trainable:
                    np_array = np.array(param._grad_ivar().value().get_tensor())
                    dy_grad_value[param.name + base.core.grad_var_suffix()] = (
                        np_array
                    )

            resnet.clear_gradients()

            dy_param_value = {}
            for param in resnet.parameters():
                dy_param_value[param.name] = param.numpy()

            if use_save_load and batch_id == 2:
                paddle.save(scaler.state_dict(), 'ResNet_model.pdparams')
                dict_load = paddle.load('ResNet_model.pdparams')
                scaler.load_state_dict(dict_load)
        if use_data_loader:
            train_reader._reset()
        return dy_out, dy_param_value, dy_grad_value

    def test_with_state_dict(self):
        def func_isinstance():
            with base.dygraph.guard():
                out_use_state_dict = self.train_resnet(
                    enable_amp=True, use_data_loader=True, use_save_load=True
                )
                out_no_state_dict = self.train_resnet(
                    enable_amp=True, use_data_loader=True, use_save_load=False
                )
            print('save_load:', out_use_state_dict[0], out_no_state_dict[0])
            np.testing.assert_allclose(
                out_use_state_dict[0], out_no_state_dict[0], rtol=1e-05
            )

        func_isinstance()


class TestAmpDecorator(unittest.TestCase):
    def test_mode_exception(self):
        def func():
            with base.dygraph.guard():
                model = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
                opt = paddle.optimizer.SGD(parameters=model.parameters())
                model, opt = paddle.amp.decorate(
                    models=model, optimizers=opt, level='O'
                )

        self.assertRaises(ValueError, func)

    def test_input_type_exception(self):
        def test_error_model():
            class MyModel:
                def __init__(self):
                    print("A fake Model")

            model = MyModel()
            with base.dygraph.guard():
                paddle.amp.decorate(models=model, optimizers=None, level='O2')

        self.assertRaises(TypeError, test_error_model)

        def test_error_distributed_model():
            model = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
            model = paddle.DataParallel(model)
            with base.dygraph.guard():
                model = paddle.amp.decorate(models=model, level='O2')

        self.assertRaises(RuntimeError, test_error_distributed_model)

        def test_error_optimizer():
            class MyOptimizer:
                def __init__(self):
                    print("A fake Optimizer")

            model = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
            opt = MyOptimizer()
            with base.dygraph.guard():
                paddle.amp.decorate(models=model, optimizers=opt, level='O2')

        self.assertRaises(TypeError, test_error_optimizer)

    def test_set_master_weight(self):
        model1 = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
        opt1 = paddle.optimizer.Adam(
            learning_rate=0.0001,
            parameters=model1.parameters(),
            multi_precision=True,
        )

        model2 = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
        opt2 = paddle.optimizer.Adam(
            learning_rate=0.0001,
            parameters=model2.parameters(),
            multi_precision=False,
        )

        model1, opt1 = paddle.amp.decorate(
            models=model1, optimizers=opt1, level='O2', master_weight=None
        )
        self.assertEqual(opt1._multi_precision, True)

        models, opt2 = paddle.amp.decorate(
            models=[model1, model2],
            optimizers=opt2,
            level='O2',
            master_weight=None,
        )
        self.assertEqual(opt2._multi_precision, True)

        model3 = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
        opt3 = paddle.optimizer.Adam(
            learning_rate=0.0001, parameters=model3.parameters()
        )

        model4 = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
        opt4 = paddle.optimizer.Adam(
            learning_rate=0.0001, parameters=model4.parameters()
        )

        model3, opts = paddle.amp.decorate(
            models=model3,
            optimizers=[opt3, opt4],
            level='O2',
            master_weight=True,
        )
        self.assertEqual(opts[0]._multi_precision, True)
        self.assertEqual(opts[1]._multi_precision, True)

        models = [model3, model4]
        optimizers = [opt3, opt4]
        models, optimizers = paddle.amp.decorate(
            models=models,
            optimizers=optimizers,
            level='O2',
            master_weight=False,
        )
        self.assertEqual(optimizers[0]._multi_precision, False)
        self.assertEqual(optimizers[1]._multi_precision, False)

    def test_skip_BatchNorm_Layer_norm(self):
        model = paddle.nn.LayerNorm(1)
        model = paddle.amp.decorate(models=model, level='O2')
        for param in model.parameters():
            self.assertEqual((param.dtype == paddle.float32), True)

        model = paddle.nn.BatchNorm(1)
        model = paddle.amp.decorate(models=model, level='O2')
        for param in model.parameters():
            self.assertEqual((param.dtype == paddle.float32), True)

        model = paddle.nn.BatchNorm1D(1)
        model = paddle.amp.decorate(models=model, level='O2')
        for param in model.parameters():
            self.assertEqual((param.dtype == paddle.float32), True)

        model = paddle.nn.BatchNorm2D(1)
        model = paddle.amp.decorate(models=model, level='O2')
        for param in model.parameters():
            self.assertEqual((param.dtype == paddle.float32), True)

        model = paddle.nn.BatchNorm3D(1)
        model = paddle.amp.decorate(models=model, level='O2')
        for param in model.parameters():
            self.assertEqual((param.dtype == paddle.float32), True)


class TestStateDictHookForAMP(unittest.TestCase):
    def test_state_dict_hook(self):
        def func_isinstance():
            paddle.seed(100)
            model = paddle.nn.Linear(2, 4)
            model = paddle.amp.decorate(
                models=model, level='O2', save_dtype='float32'
            )
            param_value_ori = {}
            for param in model.parameters():
                param_value_ori[param.name] = param.numpy()

            state_dict = model.state_dict()
            for key, value in state_dict.items():
                state_dict[key] = value.cast("float16")
            model.set_state_dict(state_dict)

            param_value_now = {}
            for param in model.parameters():
                param_value_now[param.name] = param.numpy()

            for key in param_value_ori.keys():
                print(np.equal(param_value_ori[key], param_value_now[key]))

        func_isinstance()


class TestPureFp16SaveLoad(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_dtype_exception(self):
        def func():
            paddle.disable_static()
            model = paddle.nn.Conv2D(3, 2, 3, bias_attr=False)
            opt = paddle.optimizer.SGD(parameters=model.parameters())
            paddle.amp.decorate(
                models=model, optimizers=opt, level='O2', save_dtype='int'
            )

        self.assertRaises(ValueError, func)

    def train_resnet(
        self, enable_amp=True, use_data_loader=True, use_save_load=True
    ):
        seed = 90

        batch_size = train_parameters["batch_size"]
        batch_num = 4

        paddle.seed(seed)
        paddle.framework.random._manual_program_seed(seed)

        resnet = ResNet(use_cudnn=True)
        optimizer = optimizer_setting(
            train_parameters, parameter_list=resnet.parameters()
        )
        np.random.seed(seed)
        train_reader = paddle.batch(
            paddle.dataset.flowers.train(use_xmap=False), batch_size=batch_size
        )

        dy_param_init_value = {}
        for param in resnet.parameters():
            dy_param_init_value[param.name] = param.numpy()

        program = None
        scaler = paddle.amp.GradScaler(
            enable=enable_amp, init_loss_scaling=2.0**10
        )

        if use_data_loader:
            train_reader = paddle.batch(
                reader_decorator(paddle.dataset.flowers.train(use_xmap=False)),
                batch_size=batch_size,
                drop_last=True,
            )
            train_loader = base.io.DataLoader.from_generator(
                capacity=4,
                use_double_buffer=True,
                iterable=True,
                return_list=True,
            )
            train_loader.set_sample_list_generator(train_reader)
            train_reader = train_loader

        if enable_amp:
            resnet, optimizer = paddle.amp.decorate(
                models=resnet,
                optimizers=optimizer,
                level='O2',
                save_dtype='float32',
            )

        for batch_id, data in enumerate(train_reader()):
            if batch_id >= batch_num:
                break
            if use_data_loader:
                img, label = data
            else:
                dy_x_data = np.array(
                    [x[0].reshape(3, 224, 224) for x in data]
                ).astype('float32')
                if (
                    len(np.array([x[1] for x in data]).astype('int64'))
                    != batch_size
                ):
                    continue
                y_data = (
                    np.array([x[1] for x in data])
                    .astype('int64')
                    .reshape(-1, 1)
                )

                img = paddle.to_tensor(dy_x_data)
                label = paddle.to_tensor(y_data)
            label.stop_gradient = True

            with paddle.amp.auto_cast(enable=enable_amp, level='O2'):
                out = resnet(img)

            loss = paddle.nn.functional.cross_entropy(input=out, label=label)
            loss = paddle.cast(loss, 'float32')
            avg_loss = paddle.mean(x=loss)

            dy_out = avg_loss.numpy()

            scaled_loss = scaler.scale(avg_loss)
            scaled_loss.backward()

            scaler.minimize(optimizer, scaled_loss)

            dy_grad_value = {}
            for param in resnet.parameters():
                if param.trainable:
                    np_array = np.array(param._grad_ivar().value().get_tensor())
                    dy_grad_value[param.name + base.core.grad_var_suffix()] = (
                        np_array
                    )

            resnet.clear_gradients()

            dy_param_value = {}
            for param in resnet.parameters():
                dy_param_value[param.name] = param.numpy()

            if use_save_load and batch_id == 2:
                # paddle.save
                obj = {
                    'model': resnet.state_dict(),
                    'opt': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                }
                path = os.path.join(self.temp_dir.name, 'model.pdparams')
                paddle.save(obj, path)
                # paddle.load
                obj_load = paddle.load(path)
                resnet = ResNet(use_cudnn=True)
                optimizer = optimizer_setting(
                    train_parameters, parameter_list=resnet.parameters()
                )
                resnet.set_state_dict(obj_load['model'])
                optimizer.set_state_dict(obj_load['opt'])
                scaler.load_state_dict(obj_load['scaler'])
                resnet, optimizer = paddle.amp.decorate(
                    models=resnet,
                    optimizers=optimizer,
                    level='O2',
                    save_dtype='float32',
                )

        if use_data_loader:
            train_reader._reset()
        return dy_out, dy_param_value, dy_grad_value

    def test_with_save_load(self):
        def func_isinstance():
            with base.dygraph.guard():
                out_use_save_load = self.train_resnet(
                    enable_amp=True, use_data_loader=True, use_save_load=True
                )
                out_no_save_load = self.train_resnet(
                    enable_amp=True, use_data_loader=True, use_save_load=False
                )
            print('save_load:', out_use_save_load[0], out_no_save_load[0])
            np.testing.assert_allclose(
                out_use_save_load[0], out_no_save_load[0], rtol=1e-05
            )

        func_isinstance()


class TestPureFp16InferenceSaveLoad(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def inference_save_load(self):
        BATCH_SIZE = 16
        BATCH_NUM = 4
        EPOCH_NUM = 4
        IMAGE_SIZE = 784
        CLASS_NUM = 10

        # define a random dataset
        class RandomDataset(paddle.io.Dataset):
            def __init__(self, num_samples):
                self.num_samples = num_samples

            def __getitem__(self, idx):
                image = np.random.random([IMAGE_SIZE]).astype('float32')
                label = np.random.randint(0, CLASS_NUM - 1, (1,)).astype(
                    'int64'
                )
                return image, label

            def __len__(self):
                return self.num_samples

        class LinearNet(nn.Layer):
            def __init__(self):
                super().__init__()
                self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)

            def forward(self, x):
                return self._linear(x)

        def train(layer, loader, loss_fn, opt):
            for epoch_id in range(EPOCH_NUM):
                for batch_id, (image, label) in enumerate(loader()):
                    with paddle.amp.auto_cast(
                        enable=True,
                        custom_white_list=None,
                        custom_black_list=None,
                        level='O2',
                    ):
                        out = layer(image)
                        loss = loss_fn(out, label)
                    loss.backward()
                    opt.step()
                    opt.clear_grad()

        # train
        layer = LinearNet()
        adam = paddle.optimizer.Adam(
            learning_rate=0.001,
            parameters=layer.parameters(),
            multi_precision=True,
        )
        loss_fn = nn.CrossEntropyLoss()
        layer, adam = paddle.amp.decorate(
            models=layer, optimizers=adam, save_dtype='float32'
        )
        dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
        loader = paddle.io.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            num_workers=2,
        )

        train(layer, loader, loss_fn, adam)

        # save
        path = os.path.join(self.temp_dir.name, 'example_model/linear')
        paddle.jit.save(
            layer, path, input_spec=[InputSpec(shape=[IMAGE_SIZE], name='x')]
        )

        # jit.load
        loaded_layer = paddle.jit.load(path)

        # inference
        loaded_layer.eval()
        x = np.random.randn(1, IMAGE_SIZE).astype('float32')
        x_tensor = paddle.to_tensor(x)
        pred = loaded_layer(x_tensor)

        # load_inference_model
        paddle.enable_static()
        exe = paddle.static.Executor()
        [
            inference_program,
            feed_target_names,
            fetch_targets,
        ] = paddle.static.load_inference_model(path, exe)
        tensor_img = x
        results = exe.run(
            inference_program,
            feed={feed_target_names[0]: tensor_img},
            fetch_list=fetch_targets,
        )
        print("pred.numpy()", pred.numpy())
        print("result", results[0])
        np.testing.assert_array_equal(pred.numpy(), results[0])
        paddle.disable_static()

    def test_inference_save_load(self):
        self.inference_save_load()


class TestResnet2(unittest.TestCase):
    """
    Use paddle-2.0 API
    """

    def train_resnet(
        self,
        enable_amp=True,
        level='O1',
        use_data_loader=False,
        use_param_group=False,
    ):
        seed = 90

        batch_size = train_parameters["batch_size"]
        batch_num = 10

        paddle.seed(seed)
        paddle.framework.random._manual_program_seed(seed)

        resnet = ResNet(use_cudnn=True)

        if use_param_group:
            conv_params = resnet.conv.parameters()
            other_params = []
            for p in resnet.parameters():
                contains = False
                for q in conv_params:
                    if p is q:
                        contains = True
                if not contains:
                    other_params.append(p)
            # NOTE(zhiqiu): The Membership test operations(in / not in) calls "is" and "equal",
            # see details: https://docs.python.org/3/reference/expressions.html#membership-test-operations.
            # So do not use other_params =  [p for p in resnet.parameters() if p not in conv_params]
            optimizer = paddle.optimizer.Momentum(
                parameters=[
                    {'params': conv_params, 'learning_rate': 0.01},
                    {'params': other_params, 'learning_rate': 0.001},
                ],
                multi_precision=True,
            )
        else:
            optimizer = paddle.optimizer.SGD(parameters=resnet.parameters())

        np.random.seed(seed)
        train_reader = paddle.batch(
            paddle.dataset.flowers.train(use_xmap=False), batch_size=batch_size
        )

        dy_param_init_value = {}
        for param in resnet.parameters():
            dy_param_init_value[param.name] = param.numpy()

        program = None
        scaler = paddle.amp.GradScaler(
            enable=enable_amp, init_loss_scaling=2.0**10
        )

        if use_data_loader:
            train_reader = paddle.batch(
                reader_decorator(paddle.dataset.flowers.train(use_xmap=False)),
                batch_size=batch_size,
                drop_last=True,
            )
            train_loader = base.io.DataLoader.from_generator(
                capacity=4,
                use_double_buffer=True,
                iterable=True,
                return_list=True,
            )
            train_loader.set_sample_list_generator(train_reader)
            train_reader = train_loader

        if enable_amp and (level == 'O2'):
            resnet = paddle.amp.decorate(models=resnet, level='O2')

        for batch_id, data in enumerate(train_reader()):
            if batch_id >= batch_num:
                break
            if use_data_loader:
                img, label = data
            else:
                dy_x_data = np.array(
                    [x[0].reshape(3, 224, 224) for x in data]
                ).astype('float32')
                if (
                    len(np.array([x[1] for x in data]).astype('int64'))
                    != batch_size
                ):
                    continue
                y_data = (
                    np.array([x[1] for x in data])
                    .astype('int64')
                    .reshape(-1, 1)
                )

                img = paddle.to_tensor(dy_x_data)
                label = paddle.to_tensor(y_data)
            label.stop_gradient = True

            with paddle.amp.auto_cast(enable=enable_amp, level=level):
                out = resnet(img)

            loss = paddle.nn.functional.cross_entropy(input=out, label=label)
            loss = paddle.cast(loss, 'float32')
            avg_loss = paddle.mean(x=loss)

            dy_out = avg_loss.numpy()

            scaled_loss = scaler.scale(avg_loss)
            scaled_loss.backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()

            dy_grad_value = {}
            for param in resnet.parameters():
                if param.trainable:
                    np_array = np.array(param._grad_ivar().value().get_tensor())
                    dy_grad_value[param.name + base.core.grad_var_suffix()] = (
                        np_array
                    )

            resnet.clear_gradients()

            dy_param_value = {}
            for param in resnet.parameters():
                dy_param_value[param.name] = param.numpy()
        if use_data_loader:
            train_reader._reset()
        return dy_out, dy_param_value, dy_grad_value

    def test_resnet(self):
        def func_isinstance():
            with base.dygraph.guard():
                out_fp32 = self.train_resnet(enable_amp=False)
                out_amp = self.train_resnet(enable_amp=True)
                out_pure_fp16 = self.train_resnet(enable_amp=True, level='O2')
            print(out_fp32[0], out_amp[0], out_pure_fp16[0])
            np.testing.assert_allclose(
                out_fp32[0], out_amp[0], rtol=1e-05, atol=1e-05
            )
            np.testing.assert_allclose(
                out_fp32[0], out_pure_fp16[0], rtol=1e-05, atol=0.01
            )

        func_isinstance()

    def test_with_data_loader(self):
        def func_isinstance():
            with base.dygraph.guard():
                out_fp32 = self.train_resnet(
                    enable_amp=False, use_data_loader=True
                )
                out_amp = self.train_resnet(
                    enable_amp=True, use_data_loader=True
                )
                out_pure_fp16 = self.train_resnet(
                    enable_amp=True, use_data_loader=True, level='O2'
                )
            print(out_fp32[0], out_amp[0], out_pure_fp16[0])
            np.testing.assert_allclose(
                out_fp32[0], out_amp[0], rtol=1e-05, atol=1e-05
            )
            np.testing.assert_allclose(
                out_fp32[0], out_pure_fp16[0], rtol=1e-05, atol=0.01
            )

        func_isinstance()

    def test_param_group(self):
        def func_isinstance():
            with base.dygraph.guard():
                out_fp32 = self.train_resnet(
                    enable_amp=False, use_data_loader=True, use_param_group=True
                )
                out_amp = self.train_resnet(
                    enable_amp=True, use_data_loader=True, use_param_group=True
                )
                out_pure_fp16 = self.train_resnet(
                    enable_amp=True,
                    use_data_loader=True,
                    use_param_group=True,
                    level='O2',
                )
            print(out_fp32[0], out_amp[0], out_pure_fp16[0])
            np.testing.assert_allclose(
                out_fp32[0], out_amp[0], rtol=1e-05, atol=1e-05
            )
            np.testing.assert_allclose(
                out_fp32[0], out_pure_fp16[0], rtol=1e-05, atol=0.01
            )

        func_isinstance()


class TestResnet(unittest.TestCase):
    """
    Use paddle-1.x API
    """

    def train_resnet(self, enable_amp=True, level='O1'):
        seed = 90

        batch_size = train_parameters["batch_size"]
        batch_num = 1

        with base.dygraph.guard():
            paddle.seed(seed)
            paddle.framework.random._manual_program_seed(seed)

            resnet = ResNet(use_cudnn=True)
            optimizer = optimizer_setting(
                train_parameters, parameter_list=resnet.parameters()
            )
            optimizer = paddle.optimizer.Momentum(
                parameters=resnet.parameters(), multi_precision=True
            )
            np.random.seed(seed)
            train_reader = paddle.batch(
                paddle.dataset.flowers.train(use_xmap=False),
                batch_size=batch_size,
            )

            dy_param_init_value = {}
            for param in resnet.parameters():
                dy_param_init_value[param.name] = param.numpy()

            program = None
            scaler = paddle.amp.AmpScaler(
                enable=enable_amp, init_loss_scaling=2.0**10
            )

            if enable_amp and (level == 'O2'):
                resnet, optimizer = paddle.amp.amp_decorate(
                    models=resnet, optimizers=optimizer, level='O2'
                )

            for batch_id, data in enumerate(train_reader()):
                if batch_id >= batch_num:
                    break
                dy_x_data = np.array(
                    [x[0].reshape(3, 224, 224) for x in data]
                ).astype('float32')
                if (
                    len(np.array([x[1] for x in data]).astype('int64'))
                    != batch_size
                ):
                    continue
                y_data = (
                    np.array([x[1] for x in data])
                    .astype('int64')
                    .reshape(-1, 1)
                )
                img = paddle.to_tensor(dy_x_data)
                label = paddle.to_tensor(y_data)
                label.stop_gradient = True
                with paddle.amp.amp_guard(enable=enable_amp, level=level):
                    out = resnet(img)

                loss = paddle.nn.functional.cross_entropy(
                    input=out, label=label, reduction='none', use_softmax=False
                )
                avg_loss = paddle.mean(x=loss)

                dy_out = avg_loss.numpy()

                scaled_loss = scaler.scale(avg_loss)
                scaled_loss.backward()

                scaler.minimize(optimizer, scaled_loss)

                dy_grad_value = {}
                for param in resnet.parameters():
                    if param.trainable:
                        np_array = np.array(
                            param._grad_ivar().value().get_tensor()
                        )
                        dy_grad_value[
                            param.name + base.core.grad_var_suffix()
                        ] = np_array

                resnet.clear_gradients()

                dy_param_value = {}
                for param in resnet.parameters():
                    dy_param_value[param.name] = param.numpy()

        return dy_out, dy_param_value, dy_grad_value

    def test_resnet(self):
        def func_isinstance():
            out_fp32 = self.train_resnet(enable_amp=False)
            out_amp = self.train_resnet(enable_amp=True)
            out_pure_fp16 = self.train_resnet(enable_amp=True, level='O2')
            print(out_fp32[0], out_amp[0], out_pure_fp16[0])
            np.testing.assert_allclose(
                out_fp32[0], out_amp[0], rtol=1e-05, atol=0.01
            )
            np.testing.assert_allclose(
                out_fp32[0], out_pure_fp16[0], rtol=1e-05, atol=0.1
            )

        func_isinstance()


class TestLayerNormFp16(unittest.TestCase):
    r'''layer_norm and batch_norm support mixed inputs, i.e., only input x is fp16
    and other params are fp32.
    '''

    def test_layer_norm_fp16(self):
        def func_isinstance():
            if base.is_compiled_with_cuda():
                with base.dygraph.guard(base.CUDAPlace(0)):
                    x = paddle.rand([2, 2, 2, 3])
                    layer_norm = paddle.nn.LayerNorm(x.shape[1:])
                    with paddle.amp.auto_cast(custom_white_list=['layer_norm']):
                        out = layer_norm(x)

                    self.assertTrue(out.dtype == paddle.float16)

        func_isinstance()


class TestBf16(unittest.TestCase):
    '''
    test amp for BF16
    '''

    def train(self, enable_amp=True, amp_level='O1'):
        paddle.seed(100)
        input = paddle.uniform((2, 4, 8, 8), dtype='float32', min=-1.0, max=1.0)
        conv = paddle.nn.Conv2D(4, 6, (3, 3))
        if amp_level == 'O2':
            conv = paddle.amp.decorate(
                models=conv, level=amp_level, dtype='bfloat16'
            )
        with paddle.amp.auto_cast(
            enable=enable_amp, level=amp_level, dtype='bfloat16'
        ):
            output = conv(input)
        output = output.cast('float32')
        return output.numpy()

    def test_bf16(self):
        def func_isinstance():
            if (
                base.core.is_compiled_with_cuda()
                and base.core.is_bfloat16_supported(paddle.CUDAPlace(0))
            ):
                out_fp32 = self.train(enable_amp=False)
                out_bf16_O1 = self.train(enable_amp=True, amp_level='O1')
                out_bf16_O2 = self.train(enable_amp=True, amp_level='O2')
                np.testing.assert_allclose(
                    out_fp32, out_bf16_O1, rtol=0.001, atol=0.1
                )
                np.testing.assert_allclose(
                    out_fp32, out_bf16_O2, rtol=0.001, atol=0.1
                )

        func_isinstance()


class TestAmpWithPyLayer(unittest.TestCase):
    def test_pylayer(self):
        class MyMM(PyLayer):
            @staticmethod
            def forward(ctx, a, b):
                ctx.save_for_backward(a, b)
                return a.mm(b)

            @staticmethod
            def backward(ctx, grad):
                a, b = ctx.saved_tensor()
                # NOTE(zhiqiu): a and b is float32 now, while grad is fp16 when forward runs with auto_cast()
                # thus, the mm operation raise errors because of the dtype of inputs are inconsistent before.
                return grad.mm(b.t()), a.t().mm(grad)

        x = paddle.rand([10, 10])
        y = paddle.rand([10, 10])
        x.stop_gradient = False
        y.stop_gradient = False

        # with paddle.amp.auto_cast():
        res = MyMM.apply(x, y)
        loss = paddle.mean(res)
        loss.backward()


class TestAmpWithHook(unittest.TestCase):
    def test_hook_change_dtype(self):
        def func_isinstance():
            with paddle.base.dygraph.guard():
                v = paddle.rand([3, 3])
                v.stop_gradient = False

                def foo(grad):
                    print('grad', grad, grad.dtype)  # grad's dtype is float32
                    res = paddle.mm(grad, grad)  # mm runs in fp16
                    print('res', res, res.dtype)  # res's dtype is float16
                    return res

                v.register_hook(foo)
                with paddle.amp.auto_cast():
                    a = paddle.mm(v, v)
                    loss = a.sum()
                    self.assertRaises(RuntimeError, loss.backward)

        func_isinstance()

    def test_hook_change_place(self):
        def func_isinstance():
            with paddle.base.dygraph.guard():
                v = paddle.rand([3, 3])
                v.stop_gradient = False

                def foo(grad):
                    res = grad.cpu()  # change place
                    return res

                v.register_hook(foo)
                with paddle.amp.auto_cast():
                    a = paddle.mm(v, v)
                    loss = a.sum()
                    self.assertRaises(RuntimeError, loss.backward)

        func_isinstance()


if __name__ == '__main__':
    unittest.main()
