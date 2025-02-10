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
from pass_test import PassTest

import paddle
from paddle import pir
from paddle.base import core
from paddle.pir.core import create_parameter


class TestRmsNormFusePattern(PassTest):
    r"""
     x                   x       w
     |                   |       |
    pow                  |       |
     |                   |       |
    mean     epilson     |       |
       \     /           |       |
        rsqrt            |       |
          |              |       |
            \          /         |
              multiply           |
                 |               |
                    \          /
                      multiply
    """

    def fused_rms_norm_pattern(self):
        ctx = pir.DrrPatternContext()
        pat = ctx.SourcePattern()

        def constraint_function(match_ctx):
            axis = match_ctx.VectorInt64Attr("value")
            if len(axis) > 1:
                return False
            return True

        # Source Pattern
        pow = pat.Op("pd_op.pow")

        mean = pat.Op("pd_op.mean")

        full = pat.Op("pd_op.full")

        full_int_array = pat.Op(
            "pd_op.full_int_array", {"value": pat.Attr("value")}
        )

        scale = pat.Op("pd_op.scale", {"bias": pat.Attr("bias")})

        rsqrt = pat.Op("pd_op.rsqrt")
        multiply1 = pat.Op("pd_op.multiply")
        multiply2 = pat.Op("pd_op.multiply")

        # Operation connections
        pow([pat.Tensor("x")], [pat.Tensor("pow_out")])

        full_int_array([], [pat.Tensor("full_int_array_out")])

        mean(
            [pat.Tensor("pow_out"), pat.Tensor("full_int_array_out")],
            [pat.Tensor("mean_out")],
        )

        full([], [pat.Tensor("full_out")])

        scale(
            [pat.Tensor("mean_out"), pat.Tensor("full_out")],
            [pat.Tensor("scale_out")],
        )

        rsqrt([pat.Tensor("scale_out")], [pat.Tensor("rsqrt_out")])

        multiply1(
            [pat.Tensor("rsqrt_out"), pat.Tensor("x")],
            [pat.Tensor("multiply_out1")],
        )

        multiply2(
            [pat.Tensor("multiply_out1"), pat.Tensor("w")],
            [pat.Tensor("multiply_out2")],
        )

        pat.AddConstraint(constraint_function)

        # Result Pattern
        res = pat.ResultPattern()

        def compute_begin_norm_axis(match_ctx):
            axis = match_ctx.VectorInt64Attr("value")
            pow_out_shape = match_ctx.Tensor("pow_out").shape
            return (
                len(pow_out_shape) - 1 if axis[0] == -1 else axis[0],
                "int32",
            )

        begin_norm_axis = res.ComputeAttr(compute_begin_norm_axis)

        rms_norm = res.Op(
            "pd_op.rms_norm",
            {
                "epsilon": pat.Attr("bias"),
                "begin_norm_axis": begin_norm_axis,
                "quant_scale": res.Float32Attr(-1.0),
                "quant_round_type": res.Int32Attr(0),
                "quant_max_bound": res.Float32Attr(0.0),
                "quant_min_bound": res.Float32Attr(0.0),
            },
        )

        rms_norm(
            [
                res.Tensor("x"),
                res.InputNoneTensor(),
                res.InputNoneTensor(),
                res.Tensor("w"),
                res.InputNoneTensor(),
            ],
            [
                res.Tensor("multiply_out2"),
                res.Tensor("residual_out"),
                res.Tensor("inv_var"),
            ],
        )

        return ctx

    def is_program_valid(self, program=None):
        return True

    def sample_program(self):
        fused_rms_norm_ctx = self.fused_rms_norm_pattern()
        for x_shape in [[1, 1, 4096]]:
            for w_shape in [[4096]]:
                for w_type in ['float32']:
                    for epilson in [1e-6]:
                        with paddle.pir_utils.IrGuard():
                            start_prog = paddle.static.Program()
                            main_prog = paddle.static.Program()
                            with paddle.pir.core.program_guard(
                                main_prog, start_prog
                            ):
                                x = paddle.static.data(
                                    name='x', shape=x_shape, dtype='float32'
                                )
                                w = create_parameter(
                                    name="w",
                                    shape=w_shape,
                                    dtype=w_type,
                                    initializer=paddle.nn.initializer.Assign(
                                        np.random.random(w_shape).astype(w_type)
                                    ),
                                )
                                variance = x.pow(2).mean(-1, keepdim=True)
                                x = paddle.rsqrt(variance + 1e-6) * x
                                out = x * w
                                out = paddle.assign(out)
                                self.pass_attr_list = [
                                    {
                                        'py_add_norm_fuse_pass': fused_rms_norm_ctx
                                    }
                                ]
                                self.feeds = {
                                    "x": np.random.random(x_shape).astype(
                                        "float32"
                                    ),
                                }
                                self.fetch_list = [out]
                                self.valid_op_map = {
                                    "pd_op.pow": 0,
                                    "pd_op.mean": 0,
                                    "pd_op.full": 0,
                                    "pd_op.scale": 0,
                                    "pd_op.rsqrt": 0,
                                    "pd_op.multiply": 0,
                                    "pd_op.rms_norm": 1,
                                }

                                yield [main_prog, start_prog], False

    def setUp(self):
        if core.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_check_output(self):
        self.check_pass_correct()


if __name__ == "__main__":
    unittest.main()
