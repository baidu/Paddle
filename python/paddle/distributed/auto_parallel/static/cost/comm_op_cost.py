# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
# limitations under the License

import math

import numpy as np

import paddle

from .base_cost import CommOpCost, register_op_cost


@register_op_cost
class AllreduceSumOpCost(CommOpCost):
    OP_TYPE = "c_allreduce_sum"

    def __init__(self, op=None, op_desc=None, comm_context=None):
        super().__init__(op=op, op_desc=op_desc, comm_context=comm_context)

    def calc_time(self):
        # use tree if cross machine and use ring if in a single machine
        time = None
        cluster = self.comm_context.cluster
        if not cluster.cross_machine(self.group_ranks):
            time = self.calc_time_ring()
        else:
            time = self.calc_time_tree()

        return time

    def calc_time_ring(self):
        alpha = self.comm_context.base_ring
        alpha += (
            2
            * (self.rank_count - self.machine_count)
            * self.comm_context.intra_ring
        )
        alpha += (
            2
            * (self.machine_count - 1)
            * (
                self.comm_context.inter_ring
                + self.hops * self.comm_context.switch
            )
        )
        beta = self.comm_context.get_max_beta(self.group_ranks)
        time = (
            alpha
            + 2
            * (self.rank_count - 1)
            / self.rank_count
            * self.comm_count
            * beta
        )

        return time

    def calc_time_tree(self):
        alpha = self.comm_context.base_tree
        alpha += (
            2
            * (self.rank_count / self.machine_count - 1)
            * self.comm_context.intra_tree
        )
        alpha += math.log2(self.machine_count) * (
            self.comm_context.inter_tree + self.hops * self.comm_context.switch
        )
        beta = self.comm_context.get_max_beta(self.group_ranks)

        time = alpha + 2 * self.comm_count * beta

        return time


@register_op_cost
class AllgatherOpCost(CommOpCost):
    OP_TYPE = "all_gather"

    def __init__(self, op=None, op_desc=None, comm_context=None):
        super().__init__(op=op, op_desc=op_desc, comm_context=comm_context)

    def calc_time(self):
        time = self.calc_time_ring()
        return time

    def calc_time_ring(self):
        alpha = self.comm_context.base_ring
        alpha += (
            self.rank_count - self.machine_count
        ) * self.comm_context.intra_ring
        alpha += (self.machine_count - 1) * (
            self.comm_context.inter_ring + self.hops * self.comm_context.switch
        )
        beta = self.comm_context.get_max_beta(self.group_ranks)
        time = (
            alpha
            + (self.rank_count - 1) / self.rank_count * self.comm_count * beta
        )
        return time

    @property
    def comm_count(self):
        from ..reshard import get_var_with_recursion

        if self._comm_count is None:
            dtype = None
            shape = None
            if self.op is not None:
                vars = self.op.block.vars
                try:
                    var_name = self.op.input("x")[0]
                except:
                    var_name = self.op.output("out")[0]
                var = get_var_with_recursion(
                    var_name, self.op.block, self.op.block.program
                )
                dtype = var.dtype
                shape = var.shape
            elif self.op_desc is not None:
                dtype = self.op_desc["inputs"]["X"][0][0]
                shape = self.op_desc["inputs"]["X"][0][1]

            factor = None
            if dtype == paddle.float32 or dtype == paddle.int32:
                factor = 4
            else:
                raise ValueError(f"Unsupported comm dtype {dtype}")
            comm_count = int(np.prod(shape)) * factor
            self._comm_count = comm_count

        return self._comm_count


@register_op_cost
class BroadcastOpCost(CommOpCost):
    OP_TYPE = "broadcast"

    def __init__(self, op=None, op_desc=None, comm_context=None):
        super().__init__(op=op, op_desc=op_desc, comm_context=comm_context)

    def calc_time(self):
        time = self.calc_time_ring()
        return time

    def calc_time_ring(self):
        alpha = self.comm_context.base_ring
        if self.machine_count > 1:
            alpha += (
                self.comm_context.inter_ring
                + self.hops * self.comm_context.switch
            )
        else:
            alpha += self.comm_context.intra_ring
        beta = self.comm_context.get_max_beta(self.group_ranks)
        time = alpha + self.comm_count * beta

        return time


@register_op_cost
class IdentityOpCost(CommOpCost):
    OP_TYPE = "c_identity"

    def __init__(self, op=None, op_desc=None, comm_context=None):
        super().__init__(op=op, op_desc=op_desc, comm_context=comm_context)

    def calc_time(self):
        return self.comm_count * 1 / (144 * 1e3)


@register_op_cost
class RecvOpCost(CommOpCost):
    OP_TYPE = "p_recv"

    def __init__(self, op=None, op_desc=None, comm_context=None):
        super().__init__(op=op, op_desc=op_desc, comm_context=comm_context)

    def calc_time(self):
        alpha = self.comm_context.base_ring
        if self.machine_count > 1:
            alpha += (
                self.comm_context.inter_ring
                + self.hops * self.comm_context.switch
            )
        else:
            alpha += self.comm_context.intra_ring
        beta = self.comm_context.get_max_beta(self.group_ranks)
        time = alpha + self.comm_count * beta
        return time

    @property
    def comm_count(self):
        from ..reshard import get_var_with_recursion

        if self._comm_count is None:
            dtype = None
            shape = None
            if self.op is not None:
                vars = self.op.block.vars
                try:
                    var_name = self.op.input("x")[0]
                except:
                    var_name = self.op.output("out")[0]
                var = get_var_with_recursion(
                    var_name, self.op.block, self.op.block.program
                )
                dtype = var.dtype
                shape = var.shape
            elif self.op_desc is not None:
                dtype = self.op_desc["inputs"]["X"][0][0]
                shape = self.op_desc["inputs"]["X"][0][1]

            factor = None
            if dtype == paddle.float32 or dtype == paddle.int32:
                factor = 4
            else:
                raise ValueError(f"Unsupported comm dtype {dtype}")
            comm_count = int(np.prod(shape)) * factor
            self._comm_count = comm_count

        return self._comm_count


@register_op_cost
class SendOpCost(CommOpCost):
    OP_TYPE = "p_send"

    def __init__(self, op=None, op_desc=None, comm_context=None):
        super().__init__(op=op, op_desc=op_desc, comm_context=comm_context)

    def calc_time(self):
        alpha = self.comm_context.base_ring
        if self.machine_count > 1:
            alpha += (
                self.comm_context.inter_ring
                + self.hops * self.comm_context.switch
            )
        else:
            alpha += self.comm_context.intra_ring
        beta = self.comm_context.get_max_beta(self.group_ranks)
        time = alpha + self.comm_count * beta

        return time

    @property
    def comm_count(self):
        from ..reshard import get_var_with_recursion

        if self._comm_count is None:
            dtype = None
            shape = None
            if self.op is not None:
                vars = self.op.block.vars
                try:
                    var_name = self.op.input("x")[0]
                except:
                    var_name = self.op.output("out")[0]
                var = get_var_with_recursion(
                    var_name, self.op.block, self.op.block.program
                )
                dtype = var.dtype
                shape = var.shape
            elif self.op_desc is not None:
                dtype = self.op_desc["inputs"]["X"][0][0]
                shape = self.op_desc["inputs"]["X"][0][1]

            factor = None
            if dtype == paddle.float32 or dtype == paddle.int32:
                factor = 4
            else:
                raise ValueError(f"Unsupported comm dtype {dtype}")
            comm_count = int(np.prod(shape)) * factor
            self._comm_count = comm_count

        return self._comm_count
