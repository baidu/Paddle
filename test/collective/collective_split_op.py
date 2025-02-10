# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

os.environ['FLAGS_enable_pir_api'] = '0'

from legacy_test.test_collective_base import (
    TestCollectiveRunnerBase,
    runtime_main,
)

import paddle
from paddle import base
from paddle.base import core

paddle.enable_static()


class TestCollectiveAllGather(TestCollectiveRunnerBase):
    def __init__(self):
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program):
        ring_id = 0
        nranks = 2
        with base.program_guard(main_prog, startup_program):
            tindata = paddle.static.data(
                name="tindata", shape=[-1, 10, 1000], dtype='float32'
            )
            tindata.desc.set_need_check_feed(False)
            toutdata = main_prog.current_block().create_var(
                name="outofsplit",
                dtype='float32',
                type=core.VarDesc.VarType.DENSE_TENSOR,
                persistable=False,
                stop_gradient=False,
            )
            main_prog.global_block().append_op(
                type="c_split",
                inputs={'X': tindata},
                attrs={'ring_id': ring_id, 'rank': self.rank, 'nranks': nranks},
                outputs={'Out': toutdata},
            )
            return toutdata


if __name__ == "__main__":
    runtime_main(TestCollectiveAllGather, "split", 0)
