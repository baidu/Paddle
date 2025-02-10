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

from legacy_test.test_collective_base import (
    TestCollectiveRunnerBase,
    runtime_main,
)

import paddle
from paddle import base

paddle.enable_static()


class TestCollectiveSendRecv(TestCollectiveRunnerBase):
    def __init__(self):
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program):
        ring_id = self.global_ring_id
        with base.program_guard(main_prog, startup_program):
            tindata = paddle.static.data(
                name="tindata",
                shape=[10, 1000],
                dtype='float64',
            )
            tindata.desc.set_need_check_feed(False)
            if self.rank == 0:
                main_prog.global_block().append_op(
                    type="p_send",
                    inputs={'x': tindata},
                    attrs={
                        'ring_id': ring_id,
                        'peer': 1,
                    },
                )
            else:
                main_prog.global_block().append_op(
                    type="p_recv",
                    outputs={'out': tindata},
                    attrs={
                        'peer': 0,
                        'ring_id': ring_id,
                        'dtype': tindata.dtype,
                        'out_shape': tindata.shape,
                    },
                )
            return tindata


if __name__ == "__main__":
    runtime_main(TestCollectiveSendRecv, "sendrecv", 0)
