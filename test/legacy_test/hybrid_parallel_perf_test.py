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

import random
import unittest

import numpy as np

import paddle
from paddle.distributed import fleet


def set_random_seed(seed, dp_id, rank_id):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed + dp_id)
    paddle.seed(seed + dp_id)


batch_size = 4
micro_batch_size = 2


class TestDistDPTraining(unittest.TestCase):
    def setUp(self):
        strategy = fleet.DistributedStrategy()
        self.model_parallel_size = 1
        self.data_parallel_size = 2
        self.pipeline_parallel_size = 1
        strategy.hybrid_configs = {
            "dp_degree": self.data_parallel_size,
            "mp_degree": self.model_parallel_size,
            "pp_degree": self.pipeline_parallel_size,
        }
        strategy.pipeline_configs = {
            "accumulate_steps": batch_size // micro_batch_size,
            "micro_batch_size": micro_batch_size,
        }
        fleet.init(is_collective=True, strategy=strategy)

    def build_optimizer(self, model):
        scheduler = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=[2], values=[0.001, 0.002], verbose=True
        )
        optimizer = paddle.optimizer.SGD(
            learning_rate=scheduler, parameters=model.parameters()
        )
        return scheduler, optimizer

    def test_communication_perf(self):
        test_comm_types = [
            "allreduce",
            "reduce",
            "broadcast",
            "allgather",
            "reduce_scatter",
        ]
        # test comm type in test_comm(list), scan package from 1M to 1G
        for comm_type in test_comm_types:
            fleet.collective_perf(comm_type, round=1)

        # context: {comm_type:[size, time]}
        # only test allreduce for package(1024B) and time threshold(0.00000001s),
        # and test allgather for package(8192B) and time threshold(2s),
        fleet.collective_perf("allreduce", 30, size_and_time={1024: 0.00000001})
        fleet.collective_perf("reduce", 30, size_and_time={1024: 0.00000001})
        fleet.collective_perf("broadcast", 30, size_and_time={1024: 0.00000001})
        fleet.collective_perf("allgather", 30, size_and_time={8192: 2})
        # test allreduce for specific size and time.
        fleet.collective_perf(
            "allreduce",
            round=50,
            size_and_time={1024: 0.00000001, 4096: 0.01, 8192: 2},
        )


class TestDistMPTraining(unittest.TestCase):
    def setUp(self):
        strategy = fleet.DistributedStrategy()
        self.model_parallel_size = 2
        self.data_parallel_size = 1
        self.pipeline_parallel_size = 1
        strategy.hybrid_configs = {
            "dp_degree": self.data_parallel_size,
            "mp_degree": self.model_parallel_size,
            "pp_degree": self.pipeline_parallel_size,
        }
        strategy.pipeline_configs = {
            "accumulate_steps": batch_size // micro_batch_size,
            "micro_batch_size": micro_batch_size,
        }
        fleet.init(is_collective=True, strategy=strategy)
        from paddle.distributed.fleet.base.topology import (
            CommunicateTopology,
            HybridCommunicateGroup,
        )

        topo = CommunicateTopology(
            hybrid_group_names=["data", "pipe", "sharding", "sep", "model"],
            dims=[1, 1, 1, 1, 2],
        )
        self.hcg = HybridCommunicateGroup(topo)

    def build_optimizer(self, model):
        scheduler = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=[2], values=[0.001, 0.002], verbose=True
        )
        optimizer = paddle.optimizer.SGD(
            learning_rate=scheduler, parameters=model.parameters()
        )
        return scheduler, optimizer

    def test_communication_perf(self):
        # test comm type in test_comm(list), scan package from 1M to 1G
        comm_types = ["allreduce", "allgather", "reduce_scatter"]
        for comm_type in comm_types:
            fleet.collective_perf(
                comm_type=comm_type,
                round=1,
            )
        # context: {comm_type:[size, time]}
        # only test reduce for package(1024B) and time threshold(1s),
        # and test allgather for package(8192B) and time threshold(0.00000002s)
        fleet.collective_perf("reduce", 30, size_and_time={1024: 1})
        fleet.collective_perf("allgather", 30, size_and_time={8192: 0.00000002})
        fleet.collective_perf(
            "reduce_scatter", 30, size_and_time={8192: 0.00000002}
        )
        # test allgather for specific size and time.
        fleet.collective_perf(
            "allgather",
            round=50,
            size_and_time={1024: 1, 4096: 0.01, 8192: 0.00000002},
        )


if __name__ == "__main__":
    unittest.main()
