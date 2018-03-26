#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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


class DistributedSpliter(object):
    """
    DistributedSpliter is the base class for dispatching vars
    into different pserver instance.
    You need to implement the `dispatch` inferface.
    """

    def __init__(self, pserver_endpoints):
        self._eps = pserver_endpoints

    @property
    def eps(self):
        return self._eps

    def dispatch(self, varlist):
        """
        :param varlist: a list of Variables
        :return: a map of pserver endpoint -> varname 
        """
        AssertionError("Interface has not been implemented.")


class HashName(DistributedSpliter):
    """
      Hash variable names to servral endpoints
    """

    def __init__(self, pserver_endpoints):
        super(self.__class__, self).__init__(pserver_endpoints)

    def _hash_block(self, block_str, total):
        return hash(block_str) % total

    def dispatch(self, varlist):
        eplist = []
        for var in varlist:
            server_id = self._hash_block(var.name(), len(self._eps))
            server_for_param = self._eps[server_id]
            eplist.append(server_for_param)
        return eplist


class RoundRobin(DistributedSpliter):
    """
    Distribute variables to serveral endpoints.
    """

    def __init__(self, pserver_endpoints):
        super(self.__class__, self).__init__(pserver_endpoints)
        self._pserver_idx = 0

    def dispatch(self, varlist):
        eplist = []
        for var in varlist:
            server_for_param = self._eps[self._pserver_idx]
            eplist.append(server_for_param)
            self._pserver_idx += 1
            if self._pserver_idx >= len(self._eps):
                self._pserver_idx = 0
        return eplist
