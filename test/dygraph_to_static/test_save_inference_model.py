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
from dygraph_to_static_utils import (
    Dy2StTestBase,
    test_ast_only,
    test_legacy_and_pir,
)

import paddle
from paddle import base
from paddle.autograd import PyLayer
from paddle.framework import use_pir_api
from paddle.jit.dy2static.partial_program import partial_program_from
from paddle.jit.dy2static.pir_partial_program import (
    partial_program_from as pir_partial_program_from,
)
from paddle.jit.pir_translated_layer import PIR_INFER_MODEL_SUFFIX
from paddle.jit.translated_layer import INFER_MODEL_SUFFIX, INFER_PARAMS_SUFFIX

SEED = 2020

np.random.seed(SEED)

place = paddle.CPUPlace()
if paddle.is_compiled_with_cuda():
    place = paddle.CUDAPlace(0) 
elif paddle.is_compiled_with_xpu():
    place = paddle.device.XPUPlace(0)    


class SimpleFcLayer(paddle.nn.Layer):
    def __init__(self, fc_size):
        super().__init__()
        self._linear = paddle.nn.Linear(fc_size, fc_size)

    def forward(self, x):
        y = self._linear(x)
        z = self._linear(y)
        out = paddle.mean(z)
        return out, y


class cus_tanh(PyLayer):
    @staticmethod
    def forward(ctx, x):
        y = paddle.tanh(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, dy):
        (y,) = ctx.saved_tensor()
        grad = dy * (1 - paddle.square(y))
        return grad


class SimplePyLayerNet(paddle.nn.Layer):
    def __init__(self, fc_size):
        super().__init__()
        self._linear = paddle.nn.Linear(fc_size, fc_size)

    def forward(self, x):
        y = self._linear(x)
        out = cus_tanh.apply(y)
        loss = paddle.mean(out)
        return loss, out


class TestDyToStaticSaveInferenceModel(Dy2StTestBase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    @test_ast_only
    @test_legacy_and_pir
    def test_save_inference_model(self):
        fc_size = 20
        x_data = np.random.random((fc_size, fc_size)).astype('float32')
        paddle.seed(SEED)

        x = paddle.to_tensor(x_data)
        layer = paddle.jit.to_static(SimpleFcLayer(fc_size))
        adam = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=layer.parameters()
        )

        for i in range(5):
            loss, pred = layer(x)
            loss.backward()
            adam.minimize(loss)
            layer.clear_gradients()
        # test for saving model in dygraph.guard
        infer_model_prefix = os.path.join(
            self.temp_dir.name, "test_dy2stat_inference_in_guard/model"
        )
        infer_model_dir = os.path.join(
            self.temp_dir.name, "test_dy2stat_inference_in_guard"
        )

        paddle.jit.save(
            layer=layer,
            path=infer_model_prefix,
            input_spec=[x],
            output_spec=[1] if use_pir_api() else [pred],
        )
        # Check the correctness of the inference
        dygraph_out, _ = layer(x)
        self.check_save_inference_model(layer, [x_data], dygraph_out.numpy())
        self.check_save_inference_model(
            layer,
            [x_data],
            dygraph_out.numpy(),
            fetch=[0] if use_pir_api() else [loss],
        )
        self.check_save_inference_model(
            layer, [x_data], dygraph_out.numpy(), feed=[x]
        )

    # TODO(MarioLulab): Disable PT test until we support PIR PyLayer
    @test_ast_only
    @test_legacy_and_pir
    def test_save_pylayer_model(self):
        fc_size = 20
        x_data = np.random.random((fc_size, fc_size)).astype('float32')
        paddle.framework._set_expected_place(place)

        paddle.seed(SEED)
        x = paddle.to_tensor(x_data)
        layer = paddle.jit.to_static(SimplePyLayerNet(fc_size))
        adam = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=layer.parameters()
        )

        for i in range(5):
            loss, pred = layer(x)
            loss.backward()
            adam.minimize(loss)
            layer.clear_gradients()

        infer_model_prefix = os.path.join(
            self.temp_dir.name, "test_dy2stat_inference_in_guard/model_pylayer"
        )
        paddle.jit.save(
            layer=layer,
            path=infer_model_prefix,
            input_spec=[x],
            output_spec=[1] if use_pir_api() else [pred],
        )
        # Check the correctness of the inference
        loss_out, _ = layer(x)

        loss_out_numpy = float(loss_out)
        self.check_save_inference_model(layer, [x_data], loss_out_numpy)
        self.check_save_inference_model(
            layer,
            [x_data],
            loss_out_numpy,
            fetch=[0] if use_pir_api() else [loss],
        )
        self.check_save_inference_model(
            layer, [x_data], loss_out_numpy, feed=[x]
        )

    def check_save_inference_model(
        self, model, inputs, gt_out, feed=None, fetch=None
    ):
        expected_persistable_vars = {p.name for p in model.parameters()}

        infer_model_prefix = os.path.join(
            self.temp_dir.name, "test_dy2stat_inference/model"
        )
        infer_model_dir = os.path.join(
            self.temp_dir.name, "test_dy2stat_inference"
        )
        if use_pir_api():
            model_filename = "model" + PIR_INFER_MODEL_SUFFIX
        else:
            model_filename = "model" + INFER_MODEL_SUFFIX
        params_filename = "model" + INFER_PARAMS_SUFFIX

        paddle.jit.save(
            layer=model,
            path=infer_model_prefix,
            input_spec=feed if feed else None,
            output_spec=fetch if fetch else None,
        )
        infer_out = self.load_and_run_inference(
            infer_model_dir, model_filename, params_filename, inputs
        )

        np.testing.assert_allclose(gt_out, infer_out, rtol=1e-05)

    def load_and_run_inference(
        self, model_path, model_filename, params_filename, inputs
    ):
        paddle.enable_static()
        exe = base.Executor(place)
        [
            inference_program,
            feed_target_names,
            fetch_targets,
        ] = paddle.static.io.load_inference_model(
            path_prefix=model_path,
            executor=exe,
            model_filename=model_filename,
            params_filename=params_filename,
        )
        results = exe.run(
            inference_program,
            feed=dict(zip(feed_target_names, inputs)),
            fetch_list=fetch_targets,
        )

        paddle.disable_static()
        return np.array(results[0])


class TestPartialProgramRaiseError(Dy2StTestBase):
    @test_ast_only
    def test_param_type(self):
        x_data = np.random.random((20, 20)).astype('float32')

        net = paddle.jit.to_static(SimpleFcLayer(20))
        x = paddle.to_tensor(x_data)
        out = net(x)

        program_cache = net.forward.program_cache
        _, (concrete_program, _) = program_cache.last()

        params = concrete_program.parameters

        concrete_program.parameters = params[0]
        # TypeError: Type of self._params should be list or tuple,
        # but received <class 'paddle.base.framework.EagerParamBase'>.
        with self.assertRaises(TypeError):
            if use_pir_api():
                pir_partial_program_from(concrete_program)
            else:
                partial_program_from(concrete_program)

        # Under PIR, params are tuples and cannot be modified
        if not use_pir_api():
            params[0] = "linear.w.0"
            concrete_program.parameters = params
            # TypeError: Type of self._params[0] should be framework.EagerParamBase,
            # but received <type 'str'>.
            with self.assertRaises(TypeError):
                partial_program_from(concrete_program)


if __name__ == '__main__':
    unittest.main()
