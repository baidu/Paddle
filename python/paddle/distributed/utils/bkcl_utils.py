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

import paddle


def check_bkcl_async_p2p_conn(ranks):
    ranks_per_node = int(os.getenv("XPU_PADDLE_NODES_PER_RANK", "8"))
    assert ranks_per_node > 0, (
        "The XPU_PADDLE_NODES_PER_RANK must be greater than 0, bot got %d"
        % ranks_per_node
    )
    for i in range(len(ranks) - 1):
        rank_interval = ranks[i + 1] - ranks[i]
        assert (
            rank_interval % ranks_per_node == 0
        ), "When BKCL_ASYNC_SEND_RECV is set, we assume all of the send/recv are cross-node communication, but now we found you have single node send/recv, which would cause hange problem."


def build_bkcl_async_p2p_conn(stage_id, prev_rank, next_rank, pp_comm_group):
    tmp_send_tensor = paddle.zeros([1], dtype="int32")
    if stage_id % 2 == 0:
        paddle.distributed.send(tmp_send_tensor, next_rank, pp_comm_group)
        paddle.distributed.recv(tmp_send_tensor, prev_rank, pp_comm_group)
        paddle.distributed.send(tmp_send_tensor, prev_rank, pp_comm_group)
        paddle.distributed.recv(tmp_send_tensor, next_rank, pp_comm_group)
    else:
        paddle.distributed.recv(tmp_send_tensor, prev_rank, pp_comm_group)
        paddle.distributed.send(tmp_send_tensor, next_rank, pp_comm_group)
        paddle.distributed.recv(tmp_send_tensor, next_rank, pp_comm_group)
        paddle.distributed.send(tmp_send_tensor, prev_rank, pp_comm_group)
