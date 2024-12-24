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

from legacy_test.test_collective_api_base import (
    TestCollectiveAPIRunnerBase,
    runtime_main,
)

import paddle
from paddle import base, framework
from paddle.base import data_feeder

paddle.enable_static()


def send_new(tensor, dst, group=None, sync_op=True):
    op_type = 'p_send'
    data_feeder.check_variable_and_dtype(
        tensor,
        'tensor',
        [
            'float16',
            'float32',
            'float64',
            'int32',
            'int64',
            'int8',
            'uint8',
            'bool',
            'uint16',
        ],
        op_type,
    )

    ring_id = 0 if group is None else group.id
    helper = framework.LayerHelper(op_type, **locals())
    helper.append_op(
        type=op_type,
        inputs={'x': [tensor]},
        attrs={
            'ring_id': ring_id,
            'peer': dst,
            'dynamic_shape': True,
        },
    )


def recv_new(tensor, src, group=None, sync_op=True, dtype='float32'):
    op_type = 'p_recv'
    data_feeder.check_variable_and_dtype(
        tensor,
        'tensor',
        [
            'float16',
            'float32',
            'float64',
            'int32',
            'int64',
            'int8',
            'uint8',
            'bool',
            'uint16',
        ],
        op_type,
    )
    ring_id = 0 if group is None else group.id
    helper = framework.LayerHelper(op_type, **locals())
    helper.append_op(
        type=op_type,
        outputs={'out': [tensor]},
        attrs={
            'ring_id': ring_id,
            'peer': src,
            'dynamic_shape': True,
            'out_shape': tensor.shape,
            'dtype': base.framework.convert_np_dtype_to_proto_type(dtype),
        },
    )


class TestCollectiveSendRecvAPI(TestCollectiveAPIRunnerBase):
    def __init__(self):
        self.global_ring_id = 0

    def get_model(self, main_prog, startup_program, rank, dtype="float32"):
        with base.program_guard(main_prog, startup_program):
            tindata = paddle.static.data(
                name="tindata",
                shape=[10, 1000],
                dtype=dtype,
            )
            if rank == 0:
                paddle.distributed.send(tindata, dst=1)
            else:
                paddle.distributed.recv(tindata, src=0)
            return [tindata]

    def get_model_new(
        self,
        main_prog,
        startup_program,
        rank,
        dtype='float32',
        reduce_type=None,
    ):
        with base.program_guard(main_prog, startup_program):
            tindata = paddle.static.data(
                name="tindata",
                shape=[10, 1000],
                dtype=dtype,
            )
            if rank == 0:
                send_new(tindata, dst=1)
            else:
                recv_new(tindata, src=0, dtype=dtype)
            return [tindata]


if __name__ == "__main__":
    runtime_main(TestCollectiveSendRecvAPI, "sendrecv")
