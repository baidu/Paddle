# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

"""Tests for PyLayer of Dynamic-to-Static.
Only test simple cases here."""
import sys
from pathlib import Path

from dygraph_to_static_utils import (
    enable_to_static_guard,
    to_legacy_ir_test,
    to_pir_test,
)

sys.path.append(
    str(Path(__file__).absolute().parent.parent.joinpath("legacy_test"))
)

import os
import tempfile
import unittest

import numpy as np
from test_jit_save_load import train

import paddle
from paddle.autograd.py_layer import PyLayer

SEED = 2023
np.random.seed(SEED)


def compare_result(dygraph_res, static_res, rtol=1e-5, atol=0):
    np.testing.assert_allclose(
        dygraph_res.detach().numpy(),
        static_res.detach().numpy(),
        rtol=rtol,
        atol=atol,
        err_msg=f'dygraph result is {dygraph_res}\nstatic_result is {static_res}',
    )


class scaled_layer_1(PyLayer):
    @staticmethod
    def forward(ctx, x):
        y = x * 3
        return y

    @staticmethod
    def backward(ctx, dy):
        dx = paddle.sin(dy)
        return dx


class scaled_layer_2(PyLayer):
    @staticmethod
    def forward(ctx, x1, x2):
        y = 3 * x1 + x2 / 5
        return y

    @staticmethod
    def backward(ctx, dy):
        dx1 = paddle.sin(dy)
        dx2 = paddle.cos(dy)
        return dx1, dx2


class cus_tanh_1(PyLayer):
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


class cus_tanh_2(PyLayer):
    @staticmethod
    def forward(ctx, x, func1, func2=paddle.square):
        ctx.func = func2
        y = func1(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, dy):
        (y,) = ctx.saved_tensor()
        grad = dy * (1 - ctx.func(y))
        return grad


class cus_tanh_3(PyLayer):
    @staticmethod
    def forward(ctx, x1, x2, func1, func2=paddle.square):
        y1 = func1(x1)
        y2 = func1(x2)
        ctx.save_for_backward(y1, y2)
        return 1, None, y1, y2, ''

    @staticmethod
    def backward(ctx, dy1, dy2):
        y1, y2 = ctx.saved_tensor()
        re1 = dy1 * (1 - paddle.square(y1))
        re2 = dy2 * (1 - paddle.square(y2))
        return re1, None


def user_defined_tanh(x):
    y = paddle.tanh(x)
    return y


def user_defined_square(x):
    y = paddle.square(x)
    return y


class cus_tanh_4(PyLayer):
    @staticmethod
    def forward(ctx, x, func, name="cus_tanh_4"):
        ctx.func = func
        y = user_defined_tanh(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, dy):
        (y,) = ctx.saved_tensor()
        grad = dy * (1 - ctx.func(y))
        return grad


class cus_tanh_5(PyLayer):
    @staticmethod
    def forward(ctx, x1, x2, func1, func2=paddle.square):
        ctx.func = func2
        y1 = func1(x1)
        y2 = func1(x2)
        ctx.save_for_backward(y1, y2)
        return 1, None, y1, y2, ''

    @staticmethod
    def backward(ctx, dy1, dy2):
        y1, y2 = ctx.saved_tensor()
        re1 = dy1 * (1 - ctx.func(y1))
        re2 = dy2 * (1 - paddle.square(y2))
        return re1, re2


class cus_sigmoid(PyLayer):
    @staticmethod
    def forward(ctx, x, func1, func2):
        ctx.func = func2
        y = 1 / (1 + func1(-x))
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensor()
        grad = dy * ctx.func(x) * (1 - ctx.func(x))
        return grad


class nested_layer(PyLayer):
    @staticmethod
    def forward(ctx, x1, x2):
        y = cus_tanh_1.apply(x1)
        ctx.save_for_backward(y)
        ret = y + x2
        return ret

    @staticmethod
    def backward(ctx, dy):
        (y,) = ctx.saved_tensor()
        grad1 = scaled_layer_1.apply(dy)
        grad2 = dy - paddle.square(y)
        return grad1, grad2


class SimpleNet_1(paddle.nn.Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.linear = paddle.nn.Linear(in_size, out_size)

    @paddle.jit.to_static(full_graph=True)
    def forward(self, data):
        hidden = self.linear(data)
        z = cus_tanh_1.apply(hidden)
        return z


class SimpleNet_2(paddle.nn.Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.linear = paddle.nn.Linear(in_size, out_size)

    def forward(self, x):
        y = self.linear(x)
        out = cus_tanh_2.apply(y, func1=paddle.tanh)
        return out


class SimpleNet_3(paddle.nn.Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.linear = paddle.nn.Linear(in_size, out_size)

    def forward(self, x):
        y = self.linear(x)
        out = cus_sigmoid.apply(
            y, func1=paddle.exp, func2=paddle.nn.functional.sigmoid
        )
        return out


class SimpleNetInplace(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    @paddle.jit.to_static(full_graph=True)
    def forward(self, data):
        data = data**2
        z = paddle.tanh(data)
        z = cus_tanh_1.apply(z)
        return z


class SimplePyLayerNet(paddle.nn.Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.linear = paddle.nn.Linear(in_size, out_size)

    @paddle.jit.to_static(full_graph=True)
    def forward(self, x):
        y = self.linear(x)
        out = cus_tanh_2.apply(y, func1=paddle.tanh)
        out = paddle.mean(out)
        return out


class SimplePyLayerNetMultiIn(paddle.nn.Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.linear1 = paddle.nn.Linear(in_size, out_size)
        self.linear2 = paddle.nn.Linear(in_size, out_size)

    @paddle.jit.to_static(full_graph=True)
    def forward(self, x1, x2):
        y1 = self.linear1(x1)
        y2 = self.linear1(x2)
        out = cus_tanh_2.apply(y1, paddle.tanh)
        out = out + y2
        out = paddle.mean(out)
        return out


class SimplePyLayerNetStopGrad(paddle.nn.Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.linear = paddle.nn.Linear(in_size, out_size)

    def forward(self, x):
        y = self.linear(x)
        y.stop_gradient = True
        out = cus_tanh_2.apply(y, func1=paddle.tanh)
        return out


class TestPyLayerBase(unittest.TestCase):
    def setUp(self):
        self.place = "cpu"
        if paddle.is_compiled_with_cuda():
            self.place = "gpu"
        if paddle.is_compiled_with_xpu():
            self.place = "xpu"
        self.to_static: bool = False

    def _run(self, *input_args, **input_kwargs):
        assert getattr(
            self, "dygraph_func", None
        ), "Please setting `self.dygraph_func` before calling `self._run`"

        with enable_to_static_guard(self.to_static):
            paddle.set_device(self.place)
            result = self.dygraph_func(*input_args, **input_kwargs)
            result.mean().backward()
            return result

    def _run_dygraph(self, *args, **kwargs):
        self.to_static = False
        return self._run(*args, **kwargs)

    def _run_static(self, *args, **kwargs):
        self.to_static = True
        fn = (
            to_pir_test(self._run)
            if self.run_in_pir
            else to_legacy_ir_test(self._run)
        )
        return fn(*args, **kwargs)

    # TODO(MarioLulab): In the future, this will be supported: not only `paddle.Tensor`
    # but also non-Tensor objects will be included in the argument list.
    def _run_and_compare(self, *args, **kwargs):
        # Step1. Clone args and kwargs to avoid dygraph and static overwriting with each other
        dygraph_inp_args = []
        static_inp_args = []
        for v in args:
            assert isinstance(
                v, paddle.Tensor
            ), f"Only Support `paddle.Tensor` now, but got {type(v)}"
            stop_gradient = v.stop_gradient
            # detach from the compute graph to turn `dygraph_inp_args` and `static_inp_args` into leaf nodes
            v = v.detach()
            dygraph_inp_args.append(v.clone())
            static_inp_args.append(v.clone())
            if not stop_gradient:
                dygraph_inp_args[-1].stop_gradient = False
                static_inp_args[-1].stop_gradient = False

        dygraph_inp_kwargs = {}
        static_inp_kwargs = {}
        for k, v in kwargs.items():
            stop_gradient = v.stop_gradient
            assert isinstance(
                v, paddle.Tensor
            ), "Only Support `paddle.Tensor` now"
            # detach from the compute graph to turn `dygraph_inp_kwargs` and `static_inp_kwargs` into leaf nodes
            v = v.detach()
            dygraph_inp_kwargs[k] = v.clone()
            static_inp_kwargs[k] = v.clone()
            if not stop_gradient:
                dygraph_inp_kwargs[k].stop_gradient = False
                static_inp_kwargs[k].stop_gradient = False

        # Step2. Run the dygraph and the static separately
        dygraph_res = self._run_dygraph(*dygraph_inp_args, **dygraph_inp_kwargs)
        static_res = self._run_static(*static_inp_args, **static_inp_kwargs)

        # Step3. Compare forward result between dygraph and static
        if not isinstance(dygraph_res, tuple):
            dygraph_res = (dygraph_res,)
        if not isinstance(static_res, tuple):
            static_res = (static_res,)
        for d, s in zip(dygraph_res, static_res):
            compare_result(d, s)

        # Step4. Compare grad between dygraph and static
        for i in range(len(dygraph_inp_args)):
            self.assertEqual(
                dygraph_inp_args[i].stop_gradient,
                static_inp_args[i].stop_gradient,
            )
            if dygraph_inp_args[i].stop_gradient:
                continue

            compare_result(dygraph_inp_args[i].grad, static_inp_args[i].grad)

        for key in dygraph_inp_kwargs.keys():
            self.assertEqual(
                dygraph_inp_kwargs[key].stop_gradient,
                static_inp_kwargs[key].stop_gradient,
            )
            if dygraph_inp_kwargs[key].stop_gradient:
                continue

            compare_result(
                dygraph_inp_kwargs[key].grad, static_inp_kwargs[key].grad
            )


class TestPyLayerWithoutContext(TestPyLayerBase):
    def test_single_in_single_out(self):
        @paddle.jit.to_static(full_graph=True)
        def test_func(x):
            y = scaled_layer_1.apply(x)
            return y

        self.dygraph_func = test_func

        input1 = paddle.randn([2, 3]).astype("float32")
        input1.stop_gradient = False

        self.run_in_pir = False
        self._run_and_compare(input1)

        self.run_in_pir = True
        self._run_and_compare(input1)

    def test_multi_in_single_out(self):
        @paddle.jit.to_static(full_graph=True)
        def test_func(x1, x2):
            y = scaled_layer_2.apply(x1, x2)
            return y

        self.dygraph_func = test_func

        input1 = paddle.randn([2, 3]).astype("float32")
        input2 = paddle.randn([2, 3]).astype("float32")
        input1.stop_gradient = False
        input2.stop_gradient = False

        self.run_in_pir = False
        self._run_and_compare(input1, input2)

        self.run_in_pir = True
        self._run_and_compare(input1, input2)


class TestPyLayerWithContext(TestPyLayerBase):
    def test_single_in_single_out(self):
        @paddle.jit.to_static(full_graph=True)
        def test_func(x):
            y = cus_tanh_1.apply(x)
            return y

        self.dygraph_func = test_func

        input1 = paddle.randn([2, 3]).astype("float32")
        input1.stop_gradient = False

        self.run_in_pir = False
        self._run_and_compare(input1)

        self.run_in_pir = True
        self._run_and_compare(input1)

    def test_nested_pylayer(self):
        @paddle.jit.to_static(full_graph=True)
        def test_func(x1, x2):
            y = nested_layer.apply(x1, x2)
            return y

        self.dygraph_func = test_func

        input1 = paddle.randn([2, 3]).astype("float32")
        input2 = paddle.randn([2, 3]).astype("float32")
        input1.stop_gradient = False
        input2.stop_gradient = False

        self.run_in_pir = False
        self._run_and_compare(input1, input2)

        self.run_in_pir = True
        self._run_and_compare(input1, input2)

    def test_apply_kwargs_pylayer(self):
        @paddle.jit.to_static(full_graph=True)
        def test_func(x1, x2):
            y = scaled_layer_2.apply(x1=x2, x2=x1)
            return y

        self.dygraph_func = test_func

        input1 = paddle.randn([2, 3]).astype("float32")
        input2 = paddle.randn([2, 3]).astype("float32")
        input1.stop_gradient = False
        input2.stop_gradient = False

        self.run_in_pir = False
        self._run_and_compare(input1, input2)

        self.run_in_pir = True
        self._run_and_compare(input1, input2)

    def test_non_variable_inputs(self):
        @paddle.jit.to_static(full_graph=True)
        def test_func(x):
            y = cus_tanh_2.apply(x, func1=paddle.tanh)
            return y

        self.dygraph_func = test_func

        input1 = paddle.randn([2, 3]).astype("float32")
        input1.stop_gradient = False

        self.run_in_pir = False
        self._run_and_compare(input1)

        self.run_in_pir = True
        self._run_and_compare(input1)

    def test_simple_pylayer_return_none_with_no_grad(self):
        @paddle.jit.to_static(full_graph=True)
        def test_func(input1, input2):
            z = cus_tanh_3.apply(input1, input2, paddle.tanh, paddle.square)
            z = z[2] + z[3]
            return z

        self.dygraph_func = test_func

        input1 = paddle.randn([2, 3]).astype("float32")
        input2 = paddle.randn([2, 3]).astype("float32")
        input1.stop_gradient = False
        input2.stop_gradient = True

        self.run_in_pir = False
        self._run_and_compare(input1, input2)
        self.run_in_pir = True
        self._run_and_compare(input1, input2)

    def test_simple_pylayer_return_none(self):
        @paddle.jit.to_static(full_graph=True)
        def test_func(input1, input2):
            z = cus_tanh_5.apply(input1, input2, paddle.tanh, paddle.square)
            z = z[2] + z[3]
            return z

        self.dygraph_func = test_func

        input1 = paddle.randn([2, 3]).astype("float32")
        input2 = paddle.randn([2, 3]).astype("float32")
        input1.stop_gradient = False
        input2.stop_gradient = False

        self.run_in_pir = False
        self._run_and_compare(input1, input2)

        self.run_in_pir = True
        self._run_and_compare(input1, input2)

    def test_non_variable_inputs_and_userdefined_call(self):
        @paddle.jit.to_static(full_graph=True)
        def test_func(input1):
            y = cus_tanh_4.apply(
                input1, func=user_defined_square, name="cus_tanh_test"
            )
            return y

        self.dygraph_func = test_func

        input1 = paddle.randn([2, 3]).astype("float32")
        input1.stop_gradient = False

        self.run_in_pir = False
        self._run_and_compare(input1)

        self.run_in_pir = True
        self._run_and_compare(input1)


class TestPyLayerInsideNet(TestPyLayerBase):
    def test_single_in_single_out(self):
        simple_net = SimpleNet_1(in_size=4, out_size=8)
        self.dygraph_func = simple_net

        input1 = paddle.randn([3, 4]).astype("float32")
        input1.stop_gradient = False

        self.run_in_pir = False
        self._run_and_compare(input1)

        self.run_in_pir = True
        self._run_and_compare(input1)

    def test_inplace(self):
        simple_net = SimpleNetInplace()
        self.dygraph_func = simple_net

        input1 = paddle.randn([3, 4]).astype("float32")
        input1.stop_gradient = False

        self.run_in_pir = False
        self._run_and_compare(input1)

        self.run_in_pir = True
        self._run_and_compare(input1)

    def test_non_variable_args_pylayernet(self):
        simple_net = SimplePyLayerNet(in_size=4, out_size=8)
        self.dygraph_func = simple_net

        input1 = paddle.randn([3, 4]).astype("float32")
        input1.stop_gradient = False

        self.run_in_pir = False
        self._run_and_compare(input1)

        self.run_in_pir = True
        self._run_and_compare(input1)

    def test_pylayer_net_with_no_grad(self):
        simple_net = SimplePyLayerNetMultiIn(in_size=4, out_size=8)
        self.dygraph_func = simple_net

        input1 = paddle.randn([3, 4]).astype("float32")
        input2 = paddle.randn([3, 4]).astype("float32")
        input1.stop_gradient = False
        input2.stop_gradient = True

        self.run_in_pir = False
        self._run_and_compare(input1, input2)

        self.run_in_pir = True
        self._run_and_compare(input1, input2)


class PyLayerTrainHelper(unittest.TestCase):
    def setUp(self):
        self.place = "cpu"
        if paddle.is_compiled_with_cuda():
            self.place = "gpu"
        if paddle.is_compiled_with_xpu():
            self.place = "xpu"

    def _run_train(
        self, to_static: bool, layer_builder, build_strategy=None, in_pir=True
    ):
        """
        Tests model decorated by `dygraph_to_static_output` in static graph mode. For users, the model is defined in dygraph mode and trained in static graph mode.
        """
        paddle.set_device(self.place)
        np.random.seed(SEED)
        paddle.seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)

        net = layer_builder()

        if to_static:
            net = paddle.jit.to_static(
                net, build_strategy=build_strategy, full_graph=True
            )

            train_fn = (
                to_pir_test(train) if in_pir else to_legacy_ir_test(train)
            )
            _, _, avg_loss = train_fn(net)
        else:
            _, _, avg_loss = train(net)

        return avg_loss.numpy()


class TestTrainingPyLayer(PyLayerTrainHelper):
    def test_tanh_pylayer(self):
        build_layer = lambda: SimpleNet_2(784, 20)

        legacy_static_loss = self._run_train(
            to_static=True, in_pir=False, layer_builder=build_layer
        )
        pir_static_loss = self._run_train(
            to_static=True, in_pir=True, layer_builder=build_layer
        )
        dygraph_loss = self._run_train(
            to_static=False, layer_builder=build_layer
        )

        np.testing.assert_allclose(
            legacy_static_loss,
            dygraph_loss,
            rtol=1e-05,
            err_msg=f'legacy_static_loss: {legacy_static_loss} \n dygraph_loss: {dygraph_loss}',
        )

        np.testing.assert_allclose(
            pir_static_loss,
            dygraph_loss,
            rtol=1e-05,
            err_msg=f'pir_static_loss: {pir_static_loss} \n dygraph_loss: {dygraph_loss}',
        )

    def test_sigmoid_pylayer(self):
        build_layer = lambda: SimpleNet_3(784, 20)

        legacy_static_loss = self._run_train(
            to_static=True, in_pir=False, layer_builder=build_layer
        )
        pir_static_loss = self._run_train(
            to_static=True, in_pir=True, layer_builder=build_layer
        )
        dygraph_loss = self._run_train(
            to_static=False, layer_builder=build_layer
        )

        np.testing.assert_allclose(
            legacy_static_loss,
            dygraph_loss,
            rtol=1e-05,
            err_msg=f'legacy_static_loss: {legacy_static_loss} \n dygraph_loss: {dygraph_loss}',
        )

        np.testing.assert_allclose(
            pir_static_loss,
            dygraph_loss,
            rtol=1e-05,
            err_msg=f'pir_static_loss: {pir_static_loss} \n dygraph_loss: {dygraph_loss}',
        )

    def test_pylayer_net_no_grad(self):
        build_layer = lambda: SimplePyLayerNetStopGrad(784, 20)

        legacy_static_loss = self._run_train(
            to_static=True, in_pir=False, layer_builder=build_layer
        )
        pir_static_loss = self._run_train(
            to_static=True, in_pir=True, layer_builder=build_layer
        )
        dygraph_loss = self._run_train(
            to_static=False, layer_builder=build_layer
        )

        np.testing.assert_allclose(
            legacy_static_loss,
            dygraph_loss,
            rtol=1e-05,
            err_msg=f'legacy_static_loss: {legacy_static_loss} \n dygraph_loss: {dygraph_loss}',
        )

        np.testing.assert_allclose(
            pir_static_loss,
            dygraph_loss,
            rtol=1e-05,
            err_msg=f'pir_static_loss: {pir_static_loss} \n dygraph_loss: {dygraph_loss}',
        )


class TestPyLayerJitSaveLoad(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(
            self.temp_dir.name, "test_pylayer/jit_save_model"
        )
        # enable dygraph mode
        paddle.base.enable_dygraph()
        # config seed
        paddle.seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)

    def tearDown(self):
        self.temp_dir.cleanup()

    def train_and_save_model(self, model_path=None):
        layer = SimpleNet_1(784, 20)
        example_inputs, layer, _ = train(layer)
        final_model_path = model_path if model_path else self.model_path
        orig_input_types = [type(x) for x in example_inputs]
        paddle.jit.save(
            layer=layer, path=final_model_path, input_spec=example_inputs
        )
        new_input_types = [type(x) for x in example_inputs]
        self.assertEqual(orig_input_types, new_input_types)
        return layer

    @to_legacy_ir_test
    def test_save_load(self):
        # train and save model
        train_layer = self.train_and_save_model()
        # load model
        loaded_layer = paddle.jit.load(self.model_path)
        self.load_and_inference(train_layer, loaded_layer)

    @to_pir_test
    def test_pir_save_load(self):
        # train and save model
        train_layer = self.train_and_save_model()
        # load model
        loaded_layer = paddle.jit.load(self.model_path)
        self.load_and_inference(train_layer, loaded_layer)

    def load_and_inference(self, train_layer, infer_layer):
        train_layer.eval()
        infer_layer.eval()
        # inference & compare
        x = paddle.to_tensor(np.random.random((1, 784)).astype('float32'))
        train_layer_result = train_layer(x).numpy()
        infer_layer_result = infer_layer(x).numpy()

        np.testing.assert_array_equal(train_layer_result, infer_layer_result)


class PyLayerWrongUsage(PyLayer):
    @staticmethod
    def forward(ctx, x):
        ctx.x = x
        x1 = paddle.tanh(x)
        return x1

    @staticmethod
    def backward(ctx, grad):
        x = ctx.x
        x_grad = grad * (1 - paddle.square(x))
        return x_grad


class PyLayerWrongUsageWrapper(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.layer = PyLayerWrongUsage()

    def forward(self, x):
        return PyLayerWrongUsage.apply(x)


class TestPyLayerWrongUsage(unittest.TestCase):
    def test_wrong_usage(self):
        layer = PyLayerWrongUsageWrapper()
        static_layer = paddle.jit.to_static(layer, full_graph=True)
        x = paddle.to_tensor(np.random.random((1, 784)).astype('float32'))
        with self.assertRaisesRegex(
            AttributeError,
            r"`ctx.x = tensor` is not allowed in static mode, please use `ctx.save_for_backward\(tensor\)` instead.",
        ):
            static_layer(x)


class NestedStructurePyLayer(PyLayer):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        x1 = paddle.tanh(x[0])
        y1 = paddle.tanh(x[1])
        z1 = paddle.tanh(y)
        return [x1, y1, z1]

    @staticmethod
    def backward(ctx, *grad1):
        x0, x1 = ctx.saved_tensor()
        x_grad = grad1[0] * (1 - paddle.square(x0[0]))
        y_grad = grad1[1] * (1 - paddle.square(x0[1]))
        z_grad = grad1[2] * (1 - paddle.square(x1))

        return [x_grad, y_grad], z_grad


class NestedStructurePyLayerModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.w0 = self.create_parameter(shape=[42, 42])
        self.w1 = self.create_parameter(shape=[42, 42])
        self.w2 = self.create_parameter(shape=[42, 42])

    def forward(self, x):
        y1 = paddle.matmul(x, self.w0)
        y2 = paddle.matmul(x, self.w1)
        y3 = paddle.matmul(x, self.w2)

        z = NestedStructurePyLayer.apply([y1, y2], y3)
        return z[0] + z[1] + z[2]


class TestNestedStructurePyLayer(unittest.TestCase):
    def test_nested_structure(self):
        input = paddle.randn([2, 42]).astype("float32")
        input.stop_gradient = False

        model = NestedStructurePyLayerModel()
        dygraph_res = model(input)
        dygraph_res.backward()
        dygraph_input_grads = [
            paddle.assign(input.grad),
            paddle.assign(model.w0.grad),
            paddle.assign(model.w1.grad),
            paddle.assign(model.w2.grad),
        ]
        input.clear_grad()
        model.w0.clear_grad()
        model.w1.clear_grad()
        model.w2.clear_grad()

        static_model = paddle.jit.to_static(model, full_graph=True)
        static_res = static_model(input)
        static_res.backward()
        static_input_grads = [
            paddle.assign(input.grad),
            paddle.assign(model.w0.grad),
            paddle.assign(model.w1.grad),
            paddle.assign(model.w2.grad),
        ]
        input.clear_grad()
        model.w0.clear_grad()
        model.w1.clear_grad()
        model.w2.clear_grad()
        for i, (dygraph_grad, static_grad) in enumerate(
            zip(dygraph_input_grads, static_input_grads)
        ):
            np.testing.assert_allclose(
                dygraph_grad.numpy(),
                static_grad.numpy(),
                rtol=1e-5,
                atol=0,
                err_msg=f"dygraph_grad[{i}]: {dygraph_grad} \n static_grad[{i}]: {static_grad}",
            )


class NestedStructureWithNonePyLayer(PyLayer):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        x1 = paddle.tanh(x[0])
        y1 = paddle.tanh(x[1])
        z1 = paddle.tanh(y)
        return [x1, y1, z1]

    @staticmethod
    def backward(ctx, *grad1):
        x0, x1 = ctx.saved_tensor()
        x_grad = grad1[0] * (1 - paddle.square(x0[0]))
        z_grad = grad1[2] * (1 - paddle.square(x1))

        return [x_grad, None], z_grad


class NestedStructureWithNonePyLayerModel(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.w0 = self.create_parameter(shape=[42, 42])
        self.w1 = self.create_parameter(shape=[42, 42])
        self.w2 = self.create_parameter(shape=[42, 42])

    def forward(self, x):
        y1 = paddle.matmul(x, self.w0)
        y2 = paddle.matmul(x, self.w1)
        y2.stop_gradient = True
        y3 = paddle.matmul(x, self.w2)

        z = NestedStructurePyLayer.apply([y1, y2], y3)
        return z[0] + z[1] + z[2]


class TestNestedStructureWithNonePyLayer(unittest.TestCase):
    def test_nested_structure(self):
        input = paddle.randn([2, 42]).astype("float32")
        input.stop_gradient = False

        model = NestedStructurePyLayerModel()
        dygraph_res = model(input)
        dygraph_res.backward()
        dygraph_input_grads = [
            paddle.assign(input.grad),
            paddle.assign(model.w0.grad),
            paddle.assign(model.w1.grad),
            paddle.assign(model.w2.grad),
        ]
        input.clear_grad()
        model.w0.clear_grad()
        model.w1.clear_grad()
        model.w2.clear_grad()

        static_model = paddle.jit.to_static(model, full_graph=True)
        static_res = static_model(input)
        static_res.backward()
        static_input_grads = [
            paddle.assign(input.grad),
            paddle.assign(model.w0.grad),
            paddle.assign(model.w1.grad),
            paddle.assign(model.w2.grad),
        ]
        input.clear_grad()
        model.w0.clear_grad()
        model.w1.clear_grad()
        model.w2.clear_grad()
        for i, (dygraph_grad, static_grad) in enumerate(
            zip(dygraph_input_grads, static_input_grads)
        ):
            np.testing.assert_allclose(
                dygraph_grad.numpy(),
                static_grad.numpy(),
                rtol=1e-5,
                atol=0,
                err_msg=f"dygraph_grad[{i}]: {dygraph_grad} \n static_grad[{i}]: {static_grad}",
            )


if __name__ == "__main__":
    unittest.main()
