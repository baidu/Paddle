# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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


class TestBase(unittest.TestCase):
    def setUp(self):
        # default setting
        self.net = None
        self.inputs = None
        self.input_specs = None
        self.with_prim = True
        self.with_cinn = True
        self.atol = 1e-6
        self.train_atol = 1e-6
        self.with_precision_compare = True
        self.with_train = True
        # override customized setting
        self.init()
        if self.inputs:
            self.set_input_grad()

    def set_input_grad(self):
        if self.with_train:
            for i in range(len(self.inputs)):
                if self.inputs[i].dtype in [paddle.float32, paddle.float64]:
                    self.inputs[i].stop_gradient = False

    def init(self):
        pass

    def set_flags(self):
        pass

    def train(self, net, to_static, with_prim=False, with_cinn=False):
        paddle.seed(123)
        if to_static:
            paddle.set_flags({'FLAGS_prim_all': with_prim})
            if with_cinn:
                build_strategy = paddle.static.BuildStrategy()
                build_strategy.build_cinn_pass = True
                net = paddle.jit.to_static(
                    net(),
                    build_strategy=build_strategy,
                    full_graph=True,
                    input_spec=self.input_specs,
                )
            else:
                net = paddle.jit.to_static(
                    net(), full_graph=True, input_spec=self.input_specs
                )
        if self.with_train:
            net.train()
        else:
            net.eval()
        inputs = []
        for tensor in self.inputs:
            temp_tensor = paddle.clone(tensor)
            temp_tensor.retain_grads()
            inputs.append(temp_tensor)
        inputs = tuple(inputs)
        outs = net(*inputs)
        return outs, inputs

    def test_ast_prim_cinn(self):
        if not self.net:
            return
        st_out, st_inputs = self.train(self.net, to_static=True)
        self.set_flags()
        cinn_out, cinn_inputs = self.train(
            self.net,
            to_static=True,
            with_prim=self.with_prim,
            with_cinn=self.with_cinn,
        )
        if self.with_precision_compare:
            for st, cinn in zip(
                paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)
            ):
                np.testing.assert_allclose(
                    st.numpy(), cinn.numpy(), atol=self.atol
                )
        if self.with_train:
            if isinstance(st_out, (tuple, list)):
                st_loss, cinn_loss = 0, 0
                for i in range(len(st_out)):
                    st_loss += st_out[i].mean()
                    cinn_loss += cinn_out[i].mean()
            else:
                st_loss = st_out.mean()
                cinn_loss = cinn_out.mean()
            st_loss.backward()
            st_grad = []
            for i in range(len(st_inputs)):
                if (
                    st_inputs[i].dtype != paddle.int64
                    and st_inputs[i].grad is not None
                ):
                    st_grad.append(st_inputs[i].grad.numpy().copy())
            cinn_loss.backward()
            cinn_grad = []
            for i in range(len(cinn_inputs)):
                if (
                    cinn_inputs[i].dtype != paddle.int64
                    and cinn_inputs[i].grad is not None
                ):
                    cinn_grad.append(cinn_inputs[i].grad.numpy().copy())
            for i in range(len(cinn_grad)):
                np.testing.assert_allclose(
                    st_grad[i], cinn_grad[i], atol=self.train_atol
                )
