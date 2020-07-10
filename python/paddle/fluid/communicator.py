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

# Copyright(c) 2019 PaddlePaddle Authors.All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0(the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http:  // www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .executor import global_scope
"""
Communicator is used for async distribute training in distribute_transpiler mode.
It's a wrapper of a cpp class Communicator and should be used inside fleet API.
"""
from . import core
from paddle.fluid.framework import Program
from paddle.fluid.incubate.fleet.parameter_server.mode import DistributedMode

__all__ = ['Communicator', 'LargeScaleKV']


class Communicator(object):
    def __init__(self, mode, kwargs=None, envs=None):
        """
        Communicator is used for async distribute training in distribute_transpiler mode.
        It's a wrapper of a cpp class Communicator and should be used inside fleet API.

        Args:
            program(Program): the trainers program after transpile of distribute_transpiler.
            It's used by communicator to extract the information to do communication.

        Returns:
            None

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid

                prog = fluid.Program()
                comm = fluid.communicator.Communicator(prog)
                comm.start()
                comm.stop()
        """
        # set all recv op to not_run mode

        if envs is None:
            envs = {}

        if mode == DistributedMode.SYNC:
            envs["pserver_endpoints"] = ','.join(kwargs["pserver_endpoints"])
            envs["trainer_id"] = str(kwargs["trainer_id"])

        if mode == DistributedMode.GEO:
            envs["trainers"] = str(kwargs["trainers"])
            envs["sparse_attrs"] = str(kwargs["sparse_attrs"])

        mode_str = None

        if mode == DistributedMode.SYNC:
            mode_str = "SYNC"
        elif mode == DistributedMode.ASYNC:
            mode_str = "ASYNC"
        elif mode == DistributedMode.HALF_ASYNC:
            mode_str = "HALF_ASYNC"
        elif mode == DistributedMode.GEO:
            mode_str = "GEO"

        self.mode = mode_str
        self.envs = envs
        self.communicator_ = None

    def init_with_program(self, program):
        self.communicator_ = core.DistCommunicator(self.mode, program.desc,
                                                   global_scope(), self.envs)

    def init_with_ctx(self, send_ctx, recv_ctx):
        self.communicator_ = core.DistCommunicator(self.mode, send_ctx,
                                                   recv_ctx,
                                                   global_scope(), self.envs)

    def start(self):
        """
        Start communicator. Should call before training process.

        Returns:
            None

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid

                prog = fluid.Program()
                comm = fluid.communicator.Communicator(prog)
                comm.start()
                comm.stop()
        """
        self.communicator_.start()

    def stop(self):
        """
        Stop communicator. Should call after training process.

        Returns:
            None

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid

                prog = fluid.Program()
                comm = fluid.communicator.Communicator(prog)
                comm.start()
                comm.stop()
        """
        self.communicator_.stop()

    def is_running(self):
        """
        Get communicator is running or stop.

        Returns:
            bool

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid

                prog = fluid.Program()
                comm = fluid.communicator.Communicator(prog)
                comm.is_running()
        """
        self.communicator_.is_running()

    def recv(self):
        self.communicator_.recv()


class LargeScaleKV(object):
    def __init__(self):
        self.scale_kv = core.LargeScaleKV()

    def save(self, varname, dirname):
        self.scale_kv.save(varname, dirname)

    def load(self, varname, dirname):
        self.scale_kv.load(varname, dirname)
