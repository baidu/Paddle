#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
import numpy as np
import contextlib
from framework import Program, default_main_program
from . import core

__all__ = ['Executor', 'global_scope', 'scope_guard', 'switch_scope']

g_scope = core.Scope()


def global_scope():
    return g_scope


def switch_scope(scope):
    global g_scope
    ex = g_scope
    g_scope = scope
    return ex


@contextlib.contextmanager
def scope_guard(scope):
    ex = switch_scope(scope)
    yield
    switch_scope(ex)


def as_numpy(tensor):
    if isinstance(tensor, list):
        return [as_numpy(t) for t in tensor]
    assert isinstance(tensor, core.LoDTensor)
    lod = tensor.lod()
    tensor_data = np.array(tensor)
    if len(lod) == 0:
        ans = tensor_data
    else:
        raise RuntimeError("LoD Calculate lacks unit tests and buggy")
    # elif len(lod) == 1:
    #     ans = []
    #     idx = 0
    #     while idx < len(lod) - 1:
    #         ans.append(tensor_data[lod[idx]:lod[idx + 1]])
    #         idx += 1
    # else:
    #     for l in reversed(lod):
    #         ans = []
    #         idx = 0
    #         while idx < len(l) - 1:
    #             ans.append(tensor_data[l[idx]:l[idx + 1]])
    #             idx += 1
    #         tensor_data = ans
    #     ans = tensor_data
    return ans


class Executor(object):
    def __init__(self, places):
        if not isinstance(places, list) and not isinstance(places, tuple):
            places = [places]

        act_places = []
        for each in places:
            p = core.Place()
            p.set_place(each)
            act_places.append(p)

        # TODO(dzhwinter) : only use the first place
        self.executor = core.Executor(act_places[0])
        self.places = places

    def aslodtensor(self, data):
        def accumulate(data):
            if not isinstance(data, list):
                return 1
            return sum([accumulate(sub) for sub in data])

        def parselod(data):
            seq_lens = [accumulate(seq) for seq in data]
            cur_len = 0
            lod = [cur_len]
            for l in seq_lens:
                cur_len += l
                lod.append(cur_len)
            return lod

        assert len(self.places) != 0
        if not isinstance(data, list):
            # pure tensor case
            tensor = core.LoDTensor()
            tensor.set(data, self.places[0])
            return tensor
        else:
            raise RuntimeError("Current implementation lacks unittests")
            # lodtensor case
            lod = []
            if not isinstance(data[0], list):
                lod.append(parselod(data))
                flattened_data = np.concatenate(data, axis=0).astype("int64")
            else:
                while isinstance(data[0], list):
                    lod.append(parselod(seq))
                    flattened_data = [item for seq in data for item in seq]
                    data = flattened_data
                flattened_data = np.concatenate(data, axis=0).astype("int64")
            flattened_data = flattened_data.reshape([len(flattened_data), 1])
            tensor = core.LoDTensor()
            tensor.set(flattened_data, self.places[0])
            tensor.set_lod(lod)
            return tensor

    def run(self,
            program=None,
            feed=None,
            fetch_list=None,
            feed_var_name='feed',
            fetch_var_name='fetch',
            scope=None,
            return_numpy=True):
        if feed is None:
            feed = {}
        if fetch_list is None:
            fetch_list = []

        if program is None:
            program = default_main_program()

        if not isinstance(program, Program):
            raise TypeError()

        if scope is None:
            scope = global_scope()

        program = program.clone()
        global_block = program.global_block()

        if feed_var_name in global_block.vars:
            feed_var = global_block.var(feed_var_name)
        else:
            feed_var = global_block.create_var(
                name=feed_var_name,
                type=core.VarDesc.VarType.FEED_MINIBATCH,
                persistable=True)

        if fetch_var_name in global_block.vars:
            fetch_var = global_block.var(fetch_var_name)
        else:
            fetch_var = global_block.create_var(
                name=fetch_var_name,
                type=core.VarDesc.VarType.FETCH_LIST,
                persistable=True)

        feed_count = 0
        fetch_count = 0
        for op in global_block.ops:
            if op.desc.type() == 'feed':
                feed_count += 1
                assert op.desc.input('X')[0] == feed_var_name
                name = op.desc.output('Out')[0]
                if name not in feed:
                    raise Exception("feed does not have {} variable".format(
                        name))
                cur_feed = feed[name]
                if not isinstance(cur_feed, core.LoDTensor):
                    cur_feed = self.aslodtensor(cur_feed)
                idx = op.desc.attr('col')
                core.set_feed_variable(scope, cur_feed, feed_var.name, idx)
            elif op.desc.type() == 'fetch':
                fetch_count += 1
                assert op.desc.output('Out')[0] == fetch_var_name
                name = op.desc.input('X')[0]
                if name not in [var.desc.name() for var in fetch_list]:
                    raise Exception(
                        "fetch_list does not have {} variable".format(name))
                idx = op.desc.attr('col')
                assert name == fetch_list[idx].desc.name()

        if feed_count > 0 and feed_count != len(feed):
            raise Exception(
                "Feed operators in program desc does not match 'feed'")

        if fetch_count > 0 and fetch_count != len(fetch_list):
            raise Exception(
                "Fetch operators in program desc does not match 'fetch_list'")

        if feed_count == 0:
            for i, name in enumerate(feed):
                out = global_block.var(name)
                global_block.prepend_op(
                    type='feed',
                    inputs={'X': [feed_var]},
                    outputs={'Out': [out]},
                    attrs={'col': i})
                cur_feed = feed[name]
                if not isinstance(cur_feed, core.LoDTensor):
                    cur_feed = self.aslodtensor(cur_feed)
                core.set_feed_variable(scope, cur_feed, feed_var.name, i)

        if fetch_count == 0:
            for i, var in enumerate(fetch_list):
                global_block.append_op(
                    type='fetch',
                    inputs={'X': [var]},
                    outputs={'Out': [fetch_var]},
                    attrs={'col': i})

        self.executor.run(program.desc, scope, 0, True, True)
        outs = [
            core.get_fetch_variable(scope, fetch_var_name, i)
            for i in xrange(len(fetch_list))
        ]

        if return_numpy:
            outs = as_numpy(outs)
        return outs
