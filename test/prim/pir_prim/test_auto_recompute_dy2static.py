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

import os
import unittest

import numpy as np
import parameterized as param

import paddle
from paddle.base import core

TOLERANCE = {
    "float64": {"rtol": 1e-15, "atol": 1e-15},
    "float32": {"rtol": 1e-6, "atol": 1e-6},
    "float16": {"rtol": 1e-3, "atol": 1e-3},
    "bfloat16": {"rtol": 1e-2, "atol": 1e-2},
}


def rms_norm(weight, hidden):
    variance = paddle.mean(paddle.pow(hidden, 2), axis=-1, keepdim=True)
    hidden = paddle.rsqrt(variance + 0.00001) * hidden
    return hidden * weight


class PrimNet(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, weight, hidden):
        out = rms_norm(weight, hidden)
        return out


places = ["cpu"]
if paddle.is_compiled_with_cuda():
    places.append("gpu")


@param.parameterized_class(
    ('name', 'inputs', 'dtype', 'places'),
    (
        (
            "auto_recompute_rms_norm_test1",
            [
                np.random.random(size=[4096, 4096]),
                np.random.random(size=[4096, 4096]),
            ],
            "float32",
            places,
        ),
        (
            "auto_recompute_rms_norm_test2",
            [
                np.random.random(size=[128, 256]),
                np.random.random(size=[128, 256]),
            ],
            "float32",
            places,
        ),
    ),
)
class TestDy2StaticAutoRecomputeRmsNorm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = [
            (
                x.astype(cls.dtype)
                if cls.dtype != "bfloat16"
                else x.astype("float32")
            )
            for x in cls.inputs
        ]

    def product_rms_norm_inputs(self, place):
        weight = paddle.to_tensor(self.inputs[0], dtype=self.dtype, place=place)
        hidden = paddle.to_tensor(self.inputs[1], dtype=self.dtype, place=place)
        weight.stop_gradient = False
        hidden.stop_gradient = False
        return [weight, hidden]

    def cal_rms_norm_res(self, place):
        weight, hidden = self.product_rms_norm_inputs(place)
        net = PrimNet()
        net = paddle.jit.to_static(net, full_graph=True)
        program = net.forward.get_concrete_program(weight, hidden)[
            -1
        ].program.program
        out = net(weight, hidden)
        [dweight, dhidden] = paddle.grad(out, [weight, hidden])
        return program, out, dweight, dhidden

    def prepare_run_desire_res(self):
        if os.environ.get('FLAGS_enable_auto_recompute'):
            del os.environ['FLAGS_enable_auto_recompute']
        core._set_prim_all_enabled(False)

    def prepare_run_actual_res(self):
        os.environ['FLAGS_enable_auto_recompute'] = "1"
        core._set_prim_all_enabled(True)

    def test_auto_recompute(self):
        for place in places:
            self.prepare_run_desire_res()
            res_desire = self.cal_rms_norm_res(place)

            self.prepare_run_actual_res()
            res_actual = self.cal_rms_norm_res(place)
            for desire, actual in zip(res_desire[1:], res_actual[1:]):
                np.testing.assert_allclose(
                    desire,
                    actual,
                    atol=TOLERANCE[self.dtype]["atol"],
                    rtol=TOLERANCE[self.dtype]["rtol"],
                )
            actual_program = res_actual[0]
            forward_ops = actual_program.global_block().ops[:14]
            mid_ops = actual_program.global_block().ops[14:17]
            backward_ops = actual_program.global_block().ops[17:]
            saved_values = forward_ops[10].results()[0]
            define_op = saved_values.get_defining_op()
            self.assertTrue(define_op.name() == "pd_op.rsqrt")
            for op in forward_ops:
                if op.name() == "pd_op.data":
                    continue
                op_results = op.results()
                for op_result in op_results:
                    if op_result.is_same(saved_values):
                        continue
                    else:
                        all_used_ops = op_result.all_used_ops()
                        for used_op in all_used_ops:
                            self.assertTrue(used_op in forward_ops + mid_ops)


if __name__ == '__main__':
    unittest.main()
