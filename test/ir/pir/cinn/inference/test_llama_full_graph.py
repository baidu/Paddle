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

import os
import sys
import unittest
from os.path import dirname

os.environ['FLAGS_prim_all'] = 'true'
os.environ['FLAGS_prim_enable_dynamic'] = 'true'
# os.environ['FLAGS_print_ir'] = '1'
os.environ['FLAGS_enable_pir_api'] = '1'
os.environ['FLAGS_cinn_new_cluster_op_method'] = '1'
os.environ['FLAGS_prim_forward_blacklist'] = 'pd_op.embedding'

os.environ['FLAGS_use_cinn'] = '1'
os.environ['FLAGS_dist_prim_all'] = '1'
os.environ['FLAGS_enable_auto_recompute'] = '1'

import numpy as np

import paddle

sys.path.append(dirname(dirname(__file__)))
sys.path.append("../")

import llama_test_model
import utils

paddle.enable_static()


class TestLlamaModel(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        self.config = llama_test_model.LlamaConfig(num_hidden_layers=2)
        self.input_ids = np.array(
            [
                [
                    1,
                    29871,
                    31201,
                    236,
                    138,
                    141,
                    30287,
                    30557,
                    30015,
                    233,
                    187,
                    172,
                    31969,
                    31325,
                    31043,
                    30374,
                    30024,
                ]
            ],
            dtype="int64",
        )
        self.position_ids = np.array(
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]],
            dtype="int64",
        )
        self.attention_mask = np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype="int64"
        )

    def check_jit_kernel_info(self, static_fn):
        utils.check_jit_kernel_number(static_fn, 1)
        utils.check_jit_kernel_structure(static_fn, {utils.JIT_KERNEL_NAME: 1})

    def run_static(self, mode=None):
        paddle.seed(2024)
        net = llama_test_model.LlamaModel(self.config)

        input_id = paddle.static.data(
            name="input_id", shape=[1, 17], dtype="int64"
        )
        pos_id = paddle.static.data(name="pos_id", shape=[1, 17], dtype="int64")
        att_mask = paddle.static.data(
            name="att_mask", shape=[1, 17], dtype="int64"
        )
        out = net(input_id, pos_id, att_mask)

        loss = out.sum()

        sgd = paddle.optimizer.SGD(0.1)

        sgd.minimize(loss)

        place = paddle.CUDAPlace(0)
        exe = paddle.static.Executor(place)
        main_program = paddle.pir.core.default_main_program()
        exe.run(paddle.static.default_startup_program())

        res = exe.run(
            feed={
                "input_id": self.input_ids,
                "pos_id": self.position_ids,
                "att_mask": self.attention_mask,
            },
            fetch_list=[loss, out],
        )
        ops = {op.name() for op in main_program.global_block().ops}
        ops = list(ops)
        assert "pd_op.einsum" in ops
        assert "pd_op.sum_grad" not in ops
        return res[0], np.abs(res[1]).mean()

    def test_static(self):
        ref = [90.609924, 0.16003144]
        prim_res = self.run_static(mode="prim")
        for i in range(len(ref)):
            np.testing.assert_allclose(
                ref[i],
                prim_res[i],
                rtol=1e-05,
                atol=1e-05,
                err_msg=f"***** {i}th value check failed ******",
            )


if __name__ == '__main__':
    unittest.main()
